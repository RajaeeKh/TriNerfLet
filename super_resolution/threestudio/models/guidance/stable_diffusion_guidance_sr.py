import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline,StableDiffusionUpscalePipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
import os
import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-sr-guidance")
class StableDiffusionSRGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-x4-upscaler"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = None

        noise_level: int = 20,

        use_sjc: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

        guidance_scale_sr : int = 0.0
        apply_regular_unet: bool = False

        max_step_scheduler_enabled: bool = False
        max_step_scheduler: dict = field(
            default_factory=lambda: {
                "start_step": -1,
                "total_steps": -1,
                "final_value" : -1
            }
        )

        num_inference_steps: int = 75
        apply_original_resolution:bool = True
        original_resolution_pad: bool = False
        cache_dir:Union[str, None] = None

        use_text_in_guidance: bool = False

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        if self.cfg.cache_dir is not None:
            print('cache_dir: {}'.format(self.cfg.cache_dir))
            os.makedirs(self.cfg.cache_dir,exist_ok=True)
            pipe_kwargs['cache_dir'] = self.cfg.cache_dir

        self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

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
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.low_res_scheduler = self.pipe.low_res_scheduler

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        if self.cfg.use_sjc:
            # score jacobian chaining need mu
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        threestudio.info(f"Loaded Stable Diffusion Super Resolution!")

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
        noise_level
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=noise_level
        ).sample.to(input_dtype)

    # @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs
    ):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            # input_dtype = imgs.dtype
            eps = 1e-5
            imgs = imgs.clamp(eps,1-eps) * 2.0 - 1.0
            posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
            latents = posterior.sample() * self.vae.config.scaling_factor
            return latents#.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            input_dtype = latents.dtype
            latents = F.interpolate(
                latents, (latent_height, latent_width), mode="bilinear", align_corners=False
            )
            latents = 1 / self.vae.config.scaling_factor * latents
            image = self.vae.decode(latents.to(self.weights_dtype)).sample
            image = (image * 0.5 + 0.5).clamp(0, 1)
            return image#.to(input_dtype)

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        rgb_BCHW_lr : Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = latents.shape[0]
        do_classifier_free_guidance = self.cfg.guidance_scale > 1.0
        if prompt_utils.use_perp_neg:
            raise ValueError('not nedded')
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
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

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
            # print(text_embeddings.shape)
            # predict the noise residual with unet, NO grad!
            if not do_classifier_free_guidance:
                text_embeddings = text_embeddings[1:] #uncond_text_embeddings

            batch_multiplier = 2 if do_classifier_free_guidance else 1
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latents_noisy = self.scheduler.scale_model_input(latents_noisy, t)

                # 5. Add noise to image
                noise_level = torch.tensor([self.cfg.noise_level], dtype=torch.long, device=self.device)
                # noise = randn_tensor(image.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
                noise_lr = torch.randn_like(rgb_BCHW_lr)
                image = self.low_res_scheduler.add_noise(rgb_BCHW_lr, noise_lr, noise_level)

                if (self.cfg.guidance_scale_sr > 1) and do_classifier_free_guidance:
                    # print('here')
                    text_embeddings = text_embeddings[1:] #uncond_text_embeddings
                    text_embeddings = torch.cat([text_embeddings,text_embeddings],dim=0)
                    noise_lr = torch.randn_like(rgb_BCHW_lr)
                    image2 = self.low_res_scheduler.add_noise(rgb_BCHW_lr, noise_lr, self.cfg.guidance_scale_sr*noise_level)
                    image = torch.cat([image,image2])
                    noise_level = torch.cat([noise_level,self.cfg.guidance_scale_sr*noise_level])
                else:
                    image = torch.cat([image] * batch_multiplier)
                    noise_level = torch.cat([noise_level] * image.shape[0])
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * batch_multiplier, dim=0)
                latent_model_input = torch.cat([latent_model_input, image], dim=1)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * batch_multiplier),
                    encoder_hidden_states=text_embeddings,
                    noise_level=noise_level
                )

            #TODO may need to add condition scale
            if do_classifier_free_guidance:
                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

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

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils


    def compute_regular_loss(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        rgb_BCHW_lr : Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = latents.shape[0]
        do_classifier_free_guidance = self.cfg.guidance_scale > 1.0
        if prompt_utils.use_perp_neg:
            raise ValueError('not nedded')
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
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

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
            with torch.no_grad():
                text_embeddings = prompt_utils.get_text_embeddings(
                    elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
                )
            # print(text_embeddings.shape)
            # predict the noise residual with unet, NO grad!
            if not do_classifier_free_guidance:
                text_embeddings = text_embeddings[1:] #uncond_text_embeddings

            batch_multiplier = 2 if do_classifier_free_guidance else 1

            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latents_noisy = self.scheduler.scale_model_input(latents_noisy, t)

            # 5. Add noise to image
            noise_level = torch.tensor([self.cfg.noise_level], dtype=torch.long, device=self.device)
            # noise = randn_tensor(image.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
            noise_lr = torch.randn_like(rgb_BCHW_lr)
            image = self.low_res_scheduler.add_noise(rgb_BCHW_lr, noise_lr, noise_level)

            if (self.cfg.guidance_scale_sr > 1) and do_classifier_free_guidance:
                # print('here')
                text_embeddings = text_embeddings[1:] #uncond_text_embeddings
                text_embeddings = torch.cat([text_embeddings,text_embeddings],dim=0)
                noise_lr = torch.randn_like(rgb_BCHW_lr)
                image2 = self.low_res_scheduler.add_noise(rgb_BCHW_lr, noise_lr, self.cfg.guidance_scale_sr*noise_level)
                image = torch.cat([image,image2])
                noise_level = torch.cat([noise_level,self.cfg.guidance_scale_sr*noise_level])
            else:
                image = torch.cat([image] * batch_multiplier)
                noise_level = torch.cat([noise_level] * image.shape[0])
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * batch_multiplier, dim=0)
            latent_model_input = torch.cat([latent_model_input, image], dim=1)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * batch_multiplier),
                encoder_hidden_states=text_embeddings,
                noise_level=noise_level
            )

            #TODO may need to add condition scale
            if do_classifier_free_guidance:
                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

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

        loss = w * torch.nn.functional.mse_loss(noise_pred , noise)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return loss, guidance_eval_utils

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        raise ValueError('not checked')
        batch_size = elevation.shape[0]

        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                y = latents
                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)
                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

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
                y = latents

                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)

                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

        Ds = zs - sigma * noise_pred

        if self.cfg.var_red:
            grad = -(Ds - y) / sigma
        else:
            grad = -(Ds - zs) / sigma

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": scaled_zs,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    def __call__(
        self,
        rgb_lr: Float[Tensor, "B H W C"],
        rgb_hr: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        # elevation: Float[Tensor, "B"],
        # azimuth: Float[Tensor, "B"],
        # camera_distances: Float[Tensor, "B"],
        guidance_eval=False,
        generate_hr = False,
        use_hr = True,
        **kwargs,
    ):
        batch_size = rgb_lr.shape[0]

        rgb_BCHW_lr = rgb_lr.permute(0, 3, 1, 2)
        rgb_BCHW_hr = rgb_hr.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]

        if generate_hr:
            with torch.no_grad():
                # print('*************** heereeeee 1 ***************')
                if use_hr:
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


                sr_version = self.generate_sr(rgb_BCHW_lr ,rgb_BCHW_hr,
                            prompt_utils = prompt_utils
                                              ,num_inference_steps = self.cfg.num_inference_steps
                    ,ignore_t = t)
                guidance_out = {'comp_rgb' : sr_version,'t':t.item()}
                return guidance_out

        rgb_BCHW_lr = F.interpolate(
            rgb_BCHW_lr, (128, 128), mode="bilinear", align_corners=False)
        rgb_BCHW_lr = 2*rgb_BCHW_lr-1





        rgb_BCHW_hr_512 = F.interpolate(
            rgb_BCHW_hr, (512, 512), mode="bilinear", align_corners=False
        )
        # rgb_BCHW_hr_512 = rgb_BCHW_hr

        # encode image into latents with vae
        self.vae.to(dtype=torch.float32)
        latents = self.encode_images(rgb_BCHW_hr_512)
        # print('latenst nan = {}'.format(latents.isnan().any()))

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        if self.cfg.apply_regular_unet:
            loss,guidance_eval_utils = self.compute_regular_loss(
                    latents,rgb_BCHW_lr, t, prompt_utils, torch.zeros(1), None, None
                )
            guidance_eval_utils = None
            guidance_out = {
                'loss_sds' : loss
            }
        else:
            if self.cfg.use_sjc:
                grad, guidance_eval_utils = self.compute_grad_sjc(
                    latents, t, prompt_utils, elevation, azimuth, camera_distances
                )
            else:
                grad, guidance_eval_utils = self.compute_grad_sds(
                    latents,rgb_BCHW_lr, t, prompt_utils, torch.zeros(1), None, None
                )

            grad = torch.nan_to_num(grad)
            # clip grad for stable training?
            if self.grad_clip_val is not None:
                # print('clip')
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
            raise ValueError('unimpelemnted')
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


    def generate_sr(self, rgb_lr ,rgb_hr,
                        prompt_utils
                    ,num_inference_steps = 75

                    ,ignore_t = 1000
                    ):
        batch_size = rgb_lr.shape[0]
        device = self.device
        # rgb_BCHW_lr = rgb_lr.permute(0, 3, 1, 2)
        # rgb_BCHW_hr = rgb_hr.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]

        rgb_BCHW_lr = rgb_lr
        rgb_BCHW_hr = rgb_hr


        if self.cfg.apply_original_resolution:
            if self.cfg.original_resolution_pad:
                assert max(rgb_BCHW_lr.shape[2],rgb_BCHW_lr.shape[3]) <= 128

                row_pd = 128 - rgb_BCHW_lr.shape[2]
                row_pd1 = row_pd // 2
                row_pd2 = row_pd1 + (2*row_pd1 != row_pd)
                assert (row_pd1 + row_pd2) == row_pd

                col_pd = 128 - rgb_BCHW_lr.shape[3]
                col_pd1 = col_pd // 2
                col_pd2 = col_pd1 + (2 * col_pd1 != col_pd)
                assert (col_pd1 + col_pd2) == col_pd

                rgb_BCHW_lr = F.pad(rgb_BCHW_lr,(row_pd1,row_pd2,col_pd1,col_pd2))
                assert rgb_BCHW_lr.shape[2] == 128 and rgb_BCHW_lr.shape[3] == 128
            else:
                rgb_BCHW_lr = F.interpolate(
                    rgb_BCHW_lr, (128, 128), mode="bilinear", align_corners=False)
        rgb_BCHW_lr = 2 * rgb_BCHW_lr - 1

        # 5. Add noise to image
        noise_level = torch.tensor([self.cfg.noise_level], dtype=torch.long, device=self.device)
        # noise = randn_tensor(image.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        noise = torch.randn_like(rgb_BCHW_lr)
        image = self.low_res_scheduler.add_noise(rgb_BCHW_lr, noise, noise_level)



        if self.cfg.apply_original_resolution:
            if self.cfg.original_resolution_pad:
                assert max(rgb_BCHW_hr.shape[2], rgb_BCHW_hr.shape[3]) <= 512

                row_pd = 512 - rgb_BCHW_hr.shape[2]
                row_pd1 = row_pd // 2
                row_pd2 = row_pd1 + (2 * row_pd1 != row_pd)
                assert (row_pd1 + row_pd2) == row_pd

                col_pd = 512 - rgb_BCHW_hr.shape[3]
                col_pd1 = col_pd // 2
                col_pd2 = col_pd1 + (2 * col_pd1 != col_pd)
                assert (col_pd1 + col_pd2) == col_pd

                rgb_BCHW_hr_512 = F.pad(rgb_BCHW_hr, (row_pd1, row_pd2, col_pd1, col_pd2))
                assert rgb_BCHW_hr_512.shape[2] == 512 and rgb_BCHW_hr_512.shape[3] == 512
            else:
                rgb_BCHW_hr_512 = F.interpolate(
                    rgb_BCHW_hr, (512, 512), mode="bilinear", align_corners=False
                )
        else:
            rgb_BCHW_hr_512 = rgb_BCHW_hr

        # encode image into latents with vae
        self.vae.to(dtype=torch.float32)
        image_hr = self.encode_images(rgb_BCHW_hr_512)
        # print('image_hr.isnan() = {}'.format(image_hr.isnan().any()))
        latents_noisy = torch.randn_like(image_hr) * self.scheduler.init_noise_sigma
        # print('latenst nan = {}'.format(latents.isnan().any()))

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        do_classifier_free_guidance = self.cfg.guidance_scale > 1.0
        # do_text_classifier_free_guidance = self.cfg.text_guidance_scale > 1.0

        text_embeddings = prompt_utils.get_text_embeddings(
            torch.zeros(1), None, None, self.cfg.view_dependent_prompting
        )
        # predict the noise residual with unet, NO grad!
        if not do_classifier_free_guidance:
            text_embeddings = text_embeddings[1:]  # uncond_text_embeddings

        batch_multiplier = 2 if do_classifier_free_guidance else 1
        if (self.cfg.guidance_scale_sr > 1) and do_classifier_free_guidance:
            # print('here')
            if not self.cfg.use_text_in_guidance:
                text_embeddings = text_embeddings[1:]  # uncond_text_embeddings
                text_embeddings = torch.cat([text_embeddings, text_embeddings], dim=0)
            # else:
            #     print('here')
            noise_lr = torch.randn_like(rgb_BCHW_lr)
            # image2 = self.low_res_scheduler.add_noise(rgb_BCHW_lr, noise_lr, self.cfg.guidance_scale_sr * noise_level)
            # noise_level = torch.cat([noise_level, self.cfg.guidance_scale_sr * noise_level])

            image2 = self.low_res_scheduler.add_noise(torch.zeros_like(rgb_BCHW_lr)-1, noise_lr,noise_level)
            noise_level = torch.cat([noise_level, noise_level])

            image = torch.cat([image, image2])
        else:
            image = torch.cat([image] * batch_multiplier)
            noise_level = torch.cat([noise_level] * image.shape[0])

        from tqdm import tqdm
        for i, t in enumerate(tqdm(timesteps)):
            if t > ignore_t:
                # print(t)
                noise = torch.randn_like(image_hr)
                latents_noisy = self.scheduler.add_noise(image_hr, noise, timesteps[i + 1])
                continue


            latents_noisy = self.scheduler.scale_model_input(latents_noisy, t)
            # image = torch.cat([image] * batch_multiplier)
            # noise_level = torch.cat([noise_level] * image.shape[0])



            # pred noise
            latent_model_input = torch.cat([latents_noisy] * batch_multiplier, dim=0)
            latent_model_input = torch.cat([latent_model_input, image], dim=1)
            # t = t.view(-1)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.view(-1)] * batch_multiplier),
                encoder_hidden_states=text_embeddings,
                noise_level=noise_level
            )

            # TODO may need to add condition scale
            if do_classifier_free_guidance:
                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                # noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                #         noise_pred_text - noise_pred_uncond
                # )
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )

            res = self.scheduler.step(noise_pred, t, latents_noisy)
            latents_noisy = res.prev_sample
            pred_original_sample = res.pred_original_sample

        latents = latents_noisy
        img = self.decode_latents(latents.float(),latents.shape[2],latents.shape[3])
        if self.cfg.apply_original_resolution:
            if self.cfg.original_resolution_pad:
                img = img[:,:,row_pd1:-1*row_pd2,col_pd1:-1*col_pd2]
            else:
                img =  F.interpolate(
                    img, (rgb_hr.shape[2], rgb_hr.shape[3]), mode="bilinear", align_corners=False)
        return img


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
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

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
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

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
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

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
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # print('******************')
        # print(epoch)
        # print(global_step)
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        if self.cfg.max_step_scheduler_enabled:
            if ((self.cfg.max_step_scheduler.start_step >=0) and (self.cfg.max_step_scheduler.total_steps >0)
                and (global_step >= self.cfg.max_step_scheduler.start_step)
            ):
                final_value = max(self.cfg.max_step_scheduler.final_value,min_step_percent)
                assert self.cfg.max_step_scheduler.total_steps > self.cfg.max_step_scheduler.start_step
                ratio = ((global_step - self.cfg.max_step_scheduler.start_step)
                         / (self.cfg.max_step_scheduler.total_steps - self.cfg.max_step_scheduler.start_step))
                max_step_percent = max_step_percent + ratio*(final_value - max_step_percent)
        # print('min_step_percent = {}'.format(min_step_percent))
        # print('max_step_percent = {}'.format(max_step_percent))
        self.set_min_max_steps(
            min_step_percent=min_step_percent,
            max_step_percent=max_step_percent,
        )


def test():
    from skimage.transform import rescale, resize
    import numpy as np
    import torchvision.transforms as TF

    model_id = 'stabilityai/stable-diffusion-x4-upscaler'
    # cache_dir = '/data/rajaee_data/hg_cache'
    cache_dir = None

    img_f = '/home/rajaee/datasets/nerf/ship_test/ship/train/r_28.png'
    cfg_f = 'configs/triplane-sr.yaml'
    text = ''
    targer_res = 128
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

    img = TF.Resize((targer_res, targer_res),antialias=True)(img)
    img2 = TF.Resize((4*targer_res, 4*targer_res),antialias=True)(img)

    prompt_processor = threestudio.find(cfg.prompt_processor_type)(
        cfg.prompt_processor
    )
    prompt_utils = prompt_processor()

    guidance = threestudio.find(cfg.guidance_type)(cfg.guidance)

    with torch.cuda.amp.autocast(enabled=False):
        with torch.no_grad():
            # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            # out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            hr_img = guidance.generate_sr(img, img2, num_inference_steps=75
                                          , prompt_utils=prompt_utils
                                          , ignore_t=800
                                          )

    plt.imshow(img[0].permute(1, 2, 0).clone().cpu().numpy())
    plt.show()

    plt.imshow(img2[0].permute(1, 2, 0).clone().cpu().numpy())
    plt.show()

    print(hr_img.min())
    print(hr_img.max())
    print(hr_img.shape)
    plt.imshow(hr_img[0].permute(1, 2, 0).clone().cpu().numpy())
    plt.show()

if __name__ == "__main__":
    test()