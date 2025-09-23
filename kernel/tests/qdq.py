import torch
from kernel.tests.float_utils import _float8_round, _f32_to_floatx_unpacked, pack_uint4, to_blocked, unpack_uint4, _floatx_unpacked_to_f32
from comfy_quant import tensor_absmax, quantize_per_tensor_fp8, quantize_blockwise_fp4, fp4_gemm_blockwise

F4_E2M1_MAX = 6.0
F8_E4M3_MAX = 448.0
F8_E4M3_EPS = 0.125

def nvfp4_mm(x, y, tensor_scale_x, tensor_scale_y, block_scale_x, block_scale_y, out_dtype, bias=None):
    scale_result = tensor_scale_x * tensor_scale_y
    should_add_bias_separately = bias is not None

    result = torch._scaled_mm(
        x.view(torch.float4_e2m1fn_x2),
        y.view(torch.float4_e2m1fn_x2),
        block_scale_x,
        block_scale_y,
        bias=None if should_add_bias_separately else bias,
        out_dtype=out_dtype,
        # scale_result=scale_result,  # Not supported yet
    )
    result = result * scale_result.to(out_dtype)

    if should_add_bias_separately:
        result = result + bias

    return result

def test_abs_max(input_type: torch.dtype = torch.bfloat16, num_elem: int = 1024):
    sample_tensor = torch.randn(1, num_elem, dtype=input_type, device="cuda")
    absmax = tensor_absmax(sample_tensor)
    ref = torch.max(torch.abs(sample_tensor))
    assert absmax == ref

def test_per_tensor_fp8(input_type: torch.dtype = torch.bfloat16, output_type: torch.dtype = torch.float8_e4m3fn, num_elem: int = 1024):
    sample_tensor = torch.randn(1, num_elem, dtype=input_type, device="cuda")
    lp_max = torch.finfo(output_type).max
    scale = (torch.abs(sample_tensor).max() / lp_max).to(torch.float32)

    q_tensor = quantize_per_tensor_fp8(sample_tensor, scale, output_type)

    ref = torch.clamp(sample_tensor.float() / scale.float(), -lp_max, lp_max).to(output_type)
    assert torch.allclose(q_tensor.float(), ref.float())

def pyt_quantize_nvfp4(x: torch.Tensor, per_tensor_scale: torch.Tensor):
    orig_shape = x.shape
    block_size = 16

    x = x.reshape(orig_shape[0], -1, block_size)
    max_abs = torch.amax(torch.abs(x), dim=-1)
    block_scale = max_abs / F4_E2M1_MAX
    block_scale_fp32 = block_scale.to(torch.float32)
    scaled_block_scales = block_scale_fp32 / per_tensor_scale
    scaled_block_scales_fp8 = torch.clamp(
        scaled_block_scales, min=F8_E4M3_EPS, max=F8_E4M3_MAX
    )
    scaled_block_scales_fp32 = _float8_round(scaled_block_scales_fp8)
    # We "temporarily" dequant the scaled_block_scales_fp32 to get the per_tensor_scale
    # To apply to data
    total_scale = per_tensor_scale * scaled_block_scales_fp32
    data_scaled = x / total_scale.unsqueeze(-1)
    out_scales = scaled_block_scales_fp8

    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_scaled = data_scaled.view(orig_shape)

    data_lp = _f32_to_floatx_unpacked(data_scaled, 2, 1)
    data_lp = pack_uint4(data_lp)
    return data_lp, out_scales.to(torch.float8_e4m3fn)

def fp4_x2_to_f32(a, b):
    a_u8 = unpack_uint4(a)
    b_u8 = unpack_uint4(b)

    a_f32 = _floatx_unpacked_to_f32(a_u8, 2, 1)
    b_f32 = _floatx_unpacked_to_f32(b_u8, 2, 1)
    return a_f32, b_f32

def test_quantize_blockwise_fp4(input_type: torch.dtype = torch.bfloat16, num_elem: int = 1024):
    sample_tensor = torch.randn(num_elem * 2, num_elem, dtype=input_type, device="cuda")
    lp_max = 448.0 * 6.0

    global_amax = tensor_absmax(sample_tensor)
    tensor_scale = (global_amax / lp_max).to(torch.float32)

    q_tensor, sx, qx_t, sx_t = quantize_blockwise_fp4(sample_tensor, global_amax, return_identity=True, return_transpose=False, swizzled_scale=True)
    sx = sx.flatten()

    q_ref, scale_ref = pyt_quantize_nvfp4(sample_tensor, tensor_scale)
    scale_ref = to_blocked(scale_ref)
    assert q_ref.shape == q_tensor.shape


    assert scale_ref.shape == sx.shape
    print(abs(scale_ref.float() - sx.float()).mean())

    ref_f32, k_f32 = fp4_x2_to_f32(q_ref, q_tensor)

    print(k_f32)
    print(ref_f32)

def test_nvfp4_mm():
    torch.manual_seed(0)

    x = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16).contiguous()
    w = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16).contiguous()
    bias = torch.randn(4096, device="cuda", dtype=torch.bfloat16).contiguous()

    x_absmax = tensor_absmax(x)
    w_absmax = tensor_absmax(w)

    lp_max = 448.0 * 6.0
    tensor_scale_x = x_absmax / lp_max
    tensor_scale_w = w_absmax / lp_max
    total_scale = tensor_scale_x * tensor_scale_w

    x_q_pyt, scale_ref = pyt_quantize_nvfp4(x, tensor_scale_x)
    block_scale_x_pyt = to_blocked(scale_ref, flatten=False)

    w_q_pyt, scale_ref = pyt_quantize_nvfp4(w, tensor_scale_w)
    block_scale_w_pyt = to_blocked(scale_ref, flatten=False)

    out_fp4_pyt = fp4_gemm_blockwise(x_q_pyt, block_scale_x_pyt, w_q_pyt, block_scale_w_pyt, bias=bias, block_length=16,
                                     alpha=total_scale, out_dtype=torch.bfloat16)
    out_ref = torch.nn.functional.linear(x, w, bias=bias)
    assert out_fp4_pyt.shape == out_ref.shape

    x_q, sx, _, _ = quantize_blockwise_fp4(x, x_absmax, return_identity=True,
                                                      return_transpose=False, swizzled_scale=True)
    w_q, sw, _, _ = quantize_blockwise_fp4(w, w_absmax, return_identity=True,
                                                      return_transpose=False, swizzled_scale=True)
    out_fp4 = fp4_gemm_blockwise(x_q, sx, w_q, sw, bias=bias, block_length=16, alpha=total_scale,
                       out_dtype=torch.bfloat16)

    delta_hp_pyt = torch.abs(out_fp4_pyt - out_ref).mean()
    delta_hp = torch.abs(out_fp4 - out_ref).mean()
    delta_lp = torch.abs(out_fp4 - out_fp4_pyt).mean()
    # print(delta_hp_pyt)
    # print(delta_hp)
    # print(delta_lp)
    assert torch.abs(delta_hp - delta_hp_pyt) < 1e-3
