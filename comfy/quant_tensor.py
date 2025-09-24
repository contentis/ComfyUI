import torch
from comfy_quant import tensor_absmax, quantize_per_tensor_fp8, quantize_nvfp4, fp4_gemm_blockwise

Q_TYPES = [torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2, torch.uint8]


def get_quantizer_with_constraints(target_dtype: torch.dtype):
    if target_dtype in Q_TYPES:
        q_fn = dynamic_tensor_quantizer
    else:
        raise ValueError(f"Unsupported dtype {target_dtype}")

    alignment_check_fn = lambda x: x.shape[0] % 16 or x.shape[1] % 16

    def fn(x, **kwargs):
        if alignment_check_fn(x):
            return x, None
        if x.dtype == target_dtype:
            return x, None
        return q_fn(x, dtype=target_dtype, **kwargs)

    return fn


def dynamic_tensor_quantizer(x: torch.Tensor, dtype: torch.dtype, *args, **kwargs):
    input_scale = torch.abs(x).max() / torch.finfo(dtype).max
    x = (x / input_scale).clamp(torch.finfo(dtype).min, torch.finfo(dtype).max).to(dtype=dtype)
    return x, input_scale


# def tensor_quantizer(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
#     x = (x / scale).clamp(torch.finfo(dtype).min, torch.finfo(dtype).max).to(dtype=dtype).contiguous()
#     return x, scale.float()
def tensor_quantizer(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    x = quantize_per_tensor_fp8(x, scale, dtype)
    return x, scale

def nvfp4_quantizer(x: torch.Tensor, scale: torch.Tensor, **kwargs):
    amax = scale*448.0*6.0
    x, sx = quantize_nvfp4(x, amax) # TODO currently expects amax
    return x, sx

from kernel.tests.float_utils import to_blocked
from kernel.tests.qdq import pyt_quantize_nvfp4

def nvfp4_quantizer(x: torch.Tensor, scale: torch.Tensor, **kwargs):
    w_q_pyt, scale_ref = pyt_quantize_nvfp4(x, scale)
    block_scale_w_pyt = to_blocked(scale_ref, flatten=False)
    return w_q_pyt, block_scale_w_pyt

def tensor_dequantizer(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    x = x.to(dtype=dtype) * scale.to(dtype=dtype)
    return x


def woq_fwd(self, x):
    dq_weight = self.dequantizer(self.weight, scale=self.scale_weight, dtype=x.dtype)
    bias = self.bias
    if bias is not None and bias.dtype == self.weight.dtype:
        bias = self.dequantizer(bias, torch.ones_like(self.scale_weight), x.dtype)
    return torch.nn.functional.linear(x, dq_weight, bias)

@torch.cuda.nvtx.range("quantized_fwd")
def quantized_fwd(self, input):
    tensor_2d = False
    if len(input.shape) == 2:
        tensor_2d = True
        input = input.unsqueeze(1)

    input_shape = input.shape
    input_dtype = input.dtype
    assert len(input_shape) == 3, "input must be 3D"

    scale_input = getattr(self, "scale_input", None)
    q_input, scale_input = self.quantizer(input, scale=scale_input, dtype=self.weight.dtype)
    q_input = q_input.reshape(-1, input_shape[2])
    o = torch._scaled_mm(q_input, self.weight.T, scale_a=scale_input, scale_b=self.scale_weight.float(),
                         bias=self.bias, out_dtype=input_dtype)
    if isinstance(o, tuple):
        o = o[0]
    if tensor_2d:
        return o.reshape(input_shape[0], -1)
    return o.reshape((-1, input_shape[1], self.weight.shape[0]))


@torch.cuda.nvtx.range("quantized_fwd")
def nvfp4_gemm(self, input):
    tensor_3d = False
    if len(input.shape) == 3:
        if input.shape[0] != 1:
            raise Exception("no worky")
        input = input.squeeze(0)
        tensor_3d = True # TODO batching

    input_shape = input.shape
    input_dtype = input.dtype

    q_input, block_scale_x = self.quantizer(input, scale=self.scale_input)
    # q_input = q_input.reshape(-1, input_shape[2])
    # o = torch._scaled_mm(q_input, self.weight.T, scale_a=scale_input, scale_b=self.scale_weight.float(),
    #                      bias=self.bias, out_dtype=input_dtype)
    global_scale = self.scale_input * self.scale_weight


    o = fp4_gemm_blockwise(q_input, block_scale_x, self.weight, self.block_scale_weight, bias=self.bias, block_length=16,
                           alpha=global_scale, out_dtype=input_dtype)


    if isinstance(o, tuple):
        o = o[0]
    if tensor_3d:
        o = o.unsqueeze(0)
    return o