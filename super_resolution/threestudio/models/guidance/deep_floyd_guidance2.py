from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import IFPipeline,DiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *

from einops import rearrange


# taken from: https://github.com/Picsart-AI-Research/Text2Video-Zero/blob/main/utils.py
class CrossFrameAttnProcessor:
    def __init__(self, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # Sparse Attention
        if not is_cross_attention:
            video_length = key.size()[0] // self.unet_chunk_size
            # former_frame_index = torch.arange(video_length) - 1
            # former_frame_index[0] = 0
            former_frame_index = [0] * video_length
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, former_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, former_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

#deep floyd attention
class AttnAddedKVProcessor2_0:
    def __init__(self,unet_chunk_size=2):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnAddedKVProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.unet_chunk_size = unet_chunk_size

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, out_dim=4)

        # is_cross_attention = encoder_hidden_states is not None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query, out_dim=4)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj, out_dim=4)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj, out_dim=4)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key, out_dim=4)
            value = attn.head_to_batch_dim(value, out_dim=4)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        # if not is_cross_attention:
        video_length = key.size()[0] // self.unet_chunk_size
        # former_frame_index = torch.arange(video_length) - 1
        # former_frame_index[0] = 0
        former_frame_index = [0] * video_length
        # key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key_shape = key.shape
        key = key.view(-1,video_length,*key_shape[1:])
        key = key[:, former_frame_index]
        # key = rearrange(key, "b f d c -> (b f) d c")
        key = key.view(*key_shape)
        value_shape = value.shape
        # value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = value.view(-1,video_length,*value_shape[1:])
        value = value[:, former_frame_index]
        # value = rearrange(value, "b f d c -> (b f) d c")
        value = value.view(*value_shape)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, residual.shape[1])

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states

@threestudio.register("deep-floyd-guidance2")
class DeepFloydGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "DeepFloyd/IF-I-XL-v1.0"
        # FIXME: xformers error
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = True
        guidance_scale: float = 20.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        weighting_strategy: str = "sds"

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

        num_inference_steps: int = 100
        max_step_scheduler_enabled: bool = False
        max_step_scheduler: dict = field(
            default_factory=lambda: {
                "start_step": -1,
                "total_steps": -1,
                "final_value": -1
            }
        )
        cache_dir: Union[str, None] = None
        reference_image: bool = False

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Deep Floyd ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        # Create model
        self.pipe = IFPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            text_encoder=None,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            variant="fp16" if self.cfg.half_precision_weights else None,
            torch_dtype=self.weights_dtype,
        ).to(self.device)
        # self.pipe = DiffusionPipeline.from_pretrained(
        #     self.cfg.pretrained_model_name_or_path,
        #     variant="fp16" if self.cfg.half_precision_weights else None,
        #     torch_dtype=self.weights_dtype
        # ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                threestudio.warn(
                    f"Use DeepFloyd with xformers may raise error, see https://github.com/deep-floyd/IF/issues/52 to track this problem."
                )
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        self.unet = self.pipe.unet.eval()

        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = self.pipe.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        self.reference_steps = None
        self.reference_text_embeds = None
        if self.cfg.reference_image:
            # self.attn_proc = CrossFrameAttnProcessor(unet_chunk_size=2)
            self.attn_proc = AttnAddedKVProcessor2_0(unet_chunk_size=2)
            self.unet.set_attn_processor(processor=self.attn_proc)



        threestudio.info(f"Loaded Deep Floyd!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)


    def generate_image(self,
           rgb: Float[Tensor, "B H W C"],
           prompt_utils: PromptProcessorOutput,
           elevation: Float[Tensor, "B"],
           azimuth: Float[Tensor, "B"],
           camera_distances: Float[Tensor, "B"],
           num_inference_steps=100,
           ignore_t=1000,
           final_t = 0,
           view_dependent_prompting = None,
           save_reference_steps = False,
           use_refernce_step = False
           ):
        assert prompt_utils.use_perp_neg==False
        assert ignore_t >= final_t
        assert (use_refernce_step != save_reference_steps) or (save_reference_steps == False)
        device = self.device

        neg_guidance_weights = None
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, (view_dependent_prompting if  (not (view_dependent_prompting is None))
                                else self.cfg.view_dependent_prompting)
        )
        latents_noisy = torch.randn_like(rgb) * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        do_classifier_free_guidance = self.cfg.guidance_scale > 1.0
        batch_multiplier = 2 if do_classifier_free_guidance else 1

        if not do_classifier_free_guidance:
            text_embeddings = text_embeddings[:1]

        if save_reference_steps:
            self.reference_text_embeds = text_embeddings.clone().detach()
            self.reference_steps = []

        if use_refernce_step:
            text_embeddings = torch.cat([self.reference_text_embeds,text_embeddings],dim=0)
        for i, t in enumerate(tqdm(timesteps)):
            if t > ignore_t:
                noise = torch.randn_like(rgb)
                latents_noisy = self.scheduler.add_noise(rgb, noise, timesteps[i + 1])
                continue
            if t < final_t:
                latents_noisy = pred_original_sample
                break

            # print(t)
            latents_noisy = self.scheduler.scale_model_input(latents_noisy, t)
            # image = torch.cat([image] * batch_multiplier)
            # noise_level = torch.cat([noise_level] * image.shape[0])

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * batch_multiplier, dim=0)
            if save_reference_steps:
                self.reference_steps.append(latent_model_input.clone().detach().cpu())

            t_in = torch.cat([t.view(-1)] * batch_multiplier)
            if use_refernce_step:
                latent_model_input = torch.cat([self.reference_steps[i].to(device=latent_model_input.device),latent_model_input],dim=0)
                t_in = torch.cat([t.view(-1)] * batch_multiplier * 2)
                # print(latent_model_input.shape)
            noise_pred = self.forward_unet(
                latent_model_input,
                t_in,
                encoder_hidden_states=text_embeddings,
            )
            if use_refernce_step:
                tmp , noise_pred = noise_pred.chunk(2)

            # TODO may need to add condition scale
            if do_classifier_free_guidance:
                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(latent_model_input.shape[1], dim=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(latent_model_input.shape[1], dim=1)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )
                noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            res = self.scheduler.step(noise_pred, t, latents_noisy)
            latents_noisy = res.prev_sample
            pred_original_sample = res.pred_original_sample

        image = (latents_noisy / 2 + 0.5).clamp(0, 1)
        return image

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        generate_imgs = True,
        use_img=True,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb = rgb.permute(0, 3, 1, 2)

        assert rgb_as_latents == False, f"No latent space in {self.__class__.__name__}"
        rgb = rgb * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        # latents = F.interpolate(
        #     rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
        # )

        if generate_imgs:
            with torch.no_grad():
                ref_img = None
                # print('*************** heereeeee 1 ***************')
                if self.cfg.reference_image and (self.reference_steps is None):
                    threestudio.info('******** generating reference iamge ******')
                    t = self.num_train_timesteps + torch.zeros(
                        [1],
                        dtype=torch.long,
                        device=self.device,
                    )

                    ref_img = self.generate_image(
                        rgb=torch.zeros_like(rgb),
                        prompt_utils=prompt_utils,
                        elevation=torch.zeros(1),
                        azimuth=None,
                        camera_distances=None,
                        num_inference_steps=self.cfg.num_inference_steps,
                        ignore_t=t,
                        view_dependent_prompting=False,
                        save_reference_steps=True
                    )
                    assert len(self.reference_steps) == self.cfg.num_inference_steps
                if use_img:
                    t = torch.randint(
                        self.min_step,
                        self.max_step + 1,
                        [batch_size],
                        dtype=torch.long,
                        device=self.device,
                    )
                else:
                    t = self.num_train_timesteps + torch.zeros(
                        [batch_size],
                        dtype=torch.long,
                        device=self.device,
                    )


                # sr_version = self.generate_sr(rgb_BCHW_lr ,rgb_BCHW_hr,
                #             prompt_utils = prompt_utils
                #             ,num_inference_steps = self.cfg.num_inference_steps
                #     ,ignore_t = t)
                res_img = self.generate_image(
                    rgb=rgb,
                    prompt_utils=prompt_utils,
                    elevation=elevation,
                    azimuth=azimuth,
                    camera_distances=camera_distances,
                    num_inference_steps=self.cfg.num_inference_steps,
                    ignore_t=t,
                    use_refernce_step = self.cfg.reference_image
                )
                guidance_out = {'comp_rgb' : res_img,'t':t.item(),'ref_img' : ref_img}
                return guidance_out


        raise ValueError('Unimplemented')
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 6, 64, 64)

            noise_pred_text, _ = noise_pred[:batch_size].split(3, dim=1)
            noise_pred_uncond, _ = noise_pred[batch_size : batch_size * 2].split(
                3, dim=1
            )
            noise_pred_neg, _ = noise_pred[batch_size * 2 :].split(3, dim=1)

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )  # (2B, 6, 64, 64)

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        """
        # thresholding, experimental
        if self.cfg.thresholding:
            assert batch_size == 1
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
            noise_pred = custom_ddpm_step(self.scheduler,
                noise_pred, int(t.item()), latents_noisy, **self.pipe.prepare_extra_step_kwargs(None, 0.0)
            )
        """

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if guidance_eval:
            guidance_eval_utils = {
                "use_perp_neg": prompt_utils.use_perp_neg,
                "neg_guidance_weights": neg_guidance_weights,
                "text_embeddings": text_embeddings,
                "t_orig": t,
                "latents_noisy": latents_noisy,
                "noise_pred": torch.cat([noise_pred, predicted_variance], dim=1),
            }
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]
        if use_perp_neg:
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 6, 64, 64)

            noise_pred_text, _ = noise_pred[:batch_size].split(3, dim=1)
            noise_pred_uncond, _ = noise_pred[batch_size : batch_size * 2].split(
                3, dim=1
            )
            noise_pred_neg, _ = noise_pred[batch_size * 2 :].split(3, dim=1)

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (2B, 6, 64, 64)

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return torch.cat([noise_pred, predicted_variance], dim=1)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = (latents_noisy[:bs] / 2 + 0.5).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1]
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = (latents_1step / 2 + 0.5).permute(0, 2, 3, 1)
        imgs_1orig = (pred_1orig / 2 + 0.5).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = (latents_final / 2 + 0.5).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        if self.cfg.max_step_scheduler_enabled:
            if ((self.cfg.max_step_scheduler.start_step >= 0) and (self.cfg.max_step_scheduler.total_steps > 0)
                    and (global_step >= self.cfg.max_step_scheduler.start_step)
            ):
                final_value = max(self.cfg.max_step_scheduler.final_value, min_step_percent)
                assert self.cfg.max_step_scheduler.total_steps > self.cfg.max_step_scheduler.start_step
                ratio = ((global_step - self.cfg.max_step_scheduler.start_step)
                         / (self.cfg.max_step_scheduler.total_steps - self.cfg.max_step_scheduler.start_step))
                max_step_percent = max_step_percent + ratio * (final_value - max_step_percent)
        # print('min_step_percent = {}'.format(min_step_percent))
        # print('max_step_percent = {}'.format(max_step_percent))
        self.set_min_max_steps(
            min_step_percent=min_step_percent,
            max_step_percent=max_step_percent,
        )


"""
# used by thresholding, experimental
def custom_ddpm_step(ddpm, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, generator=None, return_dict: bool = True):
    self = ddpm
    t = timestep

    prev_t = self.previous_timestep(t)

    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[t].item()
    alpha_prod_t_prev = self.alphas_cumprod[prev_t].item() if prev_t >= 0 else 1.0
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    noise_thresholded = (sample - (alpha_prod_t ** 0.5) * pred_original_sample) / (beta_prod_t ** 0.5)
    return noise_thresholded
"""

def test():
    from diffusers.utils import pt_to_pil
    import torch

    # stage 1
    stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
    # stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
    stage_1.enable_model_cpu_offload()

    # stage 2
    stage_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
    )
    # stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
    stage_2.enable_model_cpu_offload()

    # stage 3
    safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker,
                      "watermarker": stage_1.watermarker}
    stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules,
                                                torch_dtype=torch.float16)
    # stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
    stage_3.enable_model_cpu_offload()

    prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'

    # text embeds
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

    generator = torch.manual_seed(0)

    # stage 1
    image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator,
                    output_type="pt").images
    pt_to_pil(image)[0].save("./if_stage_I.png")
    pt_to_pil(image)[0].show()

    # stage 2
    image = stage_2(
        image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator,
        output_type="pt"
    ).images
    pt_to_pil(image)[0].save("./if_stage_II.png")
    pt_to_pil(image)[0].show()

    # stage 3
    image = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
    image[0].save("./if_stage_III.png")
    pt_to_pil(image)[0].show()


def test2():
    from skimage.transform import rescale, resize
    import numpy as np
    import torchvision.transforms as TF

    # torch.manual_seed(0)


    img_f = 'r_28.png'
    # cfg_f = 'configs/triplane-sr.yaml'
    cfg_f = 'configs/triplane-text_image.yaml'
    text = ''
    targer_res = 64
    device = 'cuda:0'

    from threestudio.utils.config import ExperimentConfig, load_config, parse_structured
    from threestudio.systems.base import BaseLift3DSystem
    # cfg: ExperimentConfig
    cfg: BaseLift3DSystem.Config
    cfg = load_config(cfg_f, n_gpus=1).system

    # cfg = parse_structured(BaseLift3DSystem.Config, cfg)

    import imageio
    import matplotlib.pyplot as plt
    img = imageio.imread(img_f) / 255
    img = img[:, :, :3] * img[:, :, 3:]

    # img2 = resize(img, (4 * targer_res, 4 * targer_res), anti_aliasing=True)
    # img = resize(img, (targer_res, targer_res), anti_aliasing=True)
    # img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    #
    # img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device)

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    plt.imshow(img[0].permute(1, 2, 0).clone().cpu().numpy())
    plt.show()

    img = TF.Resize((targer_res, targer_res), antialias=True)(img)
    img2 = TF.Resize((4 * targer_res, 4 * targer_res), antialias=True)(img)

    prompt_processor = threestudio.find(cfg.prompt_processor_type)(
        cfg.prompt_processor
    )
    prompt_utils = prompt_processor()

    guidance = threestudio.find(cfg.guidance_type)(cfg.guidance)

    with torch.cuda.amp.autocast(enabled=False):
        with torch.no_grad():
            # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
            # out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            # hr_img = guidance.generate_sr(img, img2, num_inference_steps=75
            #                               ,torch.zeros(1), None, None
            #                               , prompt_utils=prompt_utils
            #                               , ignore_t=800
            #                               )
            res_img = guidance.generate_image(
                rgb = img,
                prompt_utils = prompt_utils,
                elevation = torch.zeros(1) ,
                azimuth = torch.zeros(1) + 0,
                camera_distances = torch.ones(1),
                num_inference_steps = 100,
                ignore_t = 1000,
                final_t = 0
            )

    plt.imshow(img[0].permute(1, 2, 0).clone().cpu().numpy())
    plt.show()

    # plt.imshow(img2[0].permute(1, 2, 0).clone().cpu().numpy())
    # plt.show()

    print(res_img.min())
    print(res_img.max())
    print(res_img.shape)
    plt.imshow(res_img[0].permute(1, 2, 0).clone().cpu().numpy())
    plt.show()

def test3():
    from skimage.transform import rescale, resize
    import numpy as np
    import torchvision.transforms as TF

    # torch.manual_seed(0)


    img_f = 'r_28.png'
    # cfg_f = 'configs/triplane-sr.yaml'
    cfg_f = 'configs/triplane-text_image.yaml'
    text = ''
    targer_res = 64
    device = 'cuda:0'

    from threestudio.utils.config import ExperimentConfig, load_config, parse_structured
    from threestudio.systems.base import BaseLift3DSystem
    # cfg: ExperimentConfig
    cfg: BaseLift3DSystem.Config
    cfg = load_config(cfg_f, n_gpus=1).system

    # cfg = parse_structured(BaseLift3DSystem.Config, cfg)

    import imageio
    import matplotlib.pyplot as plt
    # img = imageio.imread(img_f) / 255
    # img = img[:, :, :3] * img[:, :, 3:]

    # img2 = resize(img, (4 * targer_res, 4 * targer_res), anti_aliasing=True)
    # img = resize(img, (targer_res, targer_res), anti_aliasing=True)
    # img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    #
    # img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device)

    # img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    # plt.imshow(img[0].permute(1, 2, 0).clone().cpu().numpy())
    # plt.show()

    # img = TF.Resize((targer_res, targer_res), antialias=True)(img)
    # img2 = TF.Resize((4 * targer_res, 4 * targer_res), antialias=True)(img)

    img = torch.zeros(1,3,64,64).to(device)

    prompt_processor = threestudio.find(cfg.prompt_processor_type)(
        cfg.prompt_processor
    )
    prompt_utils = prompt_processor()

    guidance = threestudio.find(cfg.guidance_type)(cfg.guidance)

    with torch.cuda.amp.autocast(enabled=False):
        with torch.no_grad():
            ref_img = guidance.generate_image(
                rgb = img,
                prompt_utils = prompt_utils,
                elevation = torch.zeros(1) ,
                azimuth = torch.zeros(1) + 0,
                camera_distances = torch.ones(1),
                num_inference_steps = 100,
                ignore_t = 1000,
                final_t = 0,
                view_dependent_prompting=False,
                save_reference_steps=True
            )
            res_img = guidance.generate_image(
                rgb=img,
                prompt_utils=prompt_utils,
                elevation=torch.zeros(1),
                azimuth=torch.zeros(1) + 0,
                camera_distances=torch.ones(1),
                num_inference_steps=100,
                ignore_t=1000,
                final_t=0,
                use_refernce_step = True
            )

    plt.imshow(img[0].permute(1, 2, 0).clone().cpu().numpy())
    plt.show()
    plt.imshow(ref_img[0].permute(1, 2, 0).clone().cpu().numpy())
    plt.show()
    plt.imshow(res_img[0].permute(1, 2, 0).clone().cpu().numpy())
    plt.show()

if __name__ == "__main__":
    # test()
    test2()
    # test3()
