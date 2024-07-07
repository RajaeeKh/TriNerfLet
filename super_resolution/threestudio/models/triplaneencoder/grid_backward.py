# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom replacement for `torch.nn.functional.grid_sample` that
supports arbitrarily high order gradients between the input and output.
Only works on 2D images and assumes
`mode='bilinear'`, `padding_mode='zeros'`, `align_corners=False`."""

import torch
from pkg_resources import parse_version

# pylint: disable=redefined-builtin
# pylint: disable=arguments-differ
# pylint: disable=protected-access

#----------------------------------------------------------------------------

enabled = True  # Enable the custom op by setting this to true.
_use_pytorch_1_11_api = parse_version(torch.__version__) >= parse_version('1.11.0a') # Allow prerelease builds of 1.11

#----------------------------------------------------------------------------

def grid_sample(input, grid,padding_mode='zeros',align_corners=False):
    if _should_use_custom_op():
        # print('hhh')
        return _GridSample2dForward.apply(input, grid,padding_mode,align_corners)
    return torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode=padding_mode, align_corners=align_corners)

#----------------------------------------------------------------------------

def _should_use_custom_op():
    return enabled

#----------------------------------------------------------------------------

class _GridSample2dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid,padding_mode,align_corners):
        assert input.ndim == 4
        assert grid.ndim == 4


        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode=padding_mode, align_corners=align_corners)
        ctx.save_for_backward(input, grid)
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        grad_input, grad_grid = _GridSample2dBackward.apply(grad_output, input, grid,padding_mode,align_corners)
        return grad_input, grad_grid,None,None

#----------------------------------------------------------------------------

class _GridSample2dBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid,padding_mode,align_corners):
        op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')[0]
        if padding_mode == 'zeros':
            padding_mode_aux = 0
        elif padding_mode == "border":
            padding_mode_aux = 1
        else:  # padding_mode == 'reflection'
            padding_mode_aux = 2

        if _use_pytorch_1_11_api:
            output_mask = (ctx.needs_input_grad[1], ctx.needs_input_grad[2])
            grad_input, grad_grid = op(grad_output, input, grid, 0, padding_mode_aux, align_corners, output_mask)
        else:
            grad_input, grad_grid = op(grad_output, input, grid, 0, padding_mode_aux, align_corners)
        ctx.save_for_backward(grid)
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        _ = grad2_grad_grid # unused
        grid, = ctx.saved_tensors
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        grad2_grad_output = None
        grad2_input = None
        grad2_grid = None

        if ctx.needs_input_grad[0]:
            grad2_grad_output = _GridSample2dForward.apply(grad2_grad_input, grid,padding_mode,align_corners)

        # assert not ctx.needs_input_grad[2]
        return grad2_grad_output, grad2_input, grad2_grid,None,None

#----------------------------------------------------------------------------

if __name__ == '__main__':
    from torch.autograd import gradcheck
    device = 'cuda:0'
    input = tuple([torch.randn(2, 3, 50, 50, dtype=torch.double, requires_grad=True, device=device) for i in range(1)])
    grid = torch.rand(2, 1, 1000, 2, dtype=torch.double, device=device)
    # grid_sample((input, grid))
    grid_smaple_func = lambda input: grid_sample(input, grid,padding_mode='border',align_corners=True)
    a = grid_smaple_func(input[0]).sum()
    a.backward()
    print(input[0].grad.abs().mean())

    test = gradcheck(grid_smaple_func, input, eps=1e-6, atol=1e-4)
    print(test)