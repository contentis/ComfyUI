import math
from typing import Tuple, Optional

import torch
import comfy_quant.ext

torch.manual_seed(0)

__all__ = [
    "tensor_absmax",
    "quantize_per_tensor_fp8",
    "quantize_nvfp4",
    "quantize_transpose_vector_blockwise_fp4",
    "fp4_gemm_blockwise",
]

_cublas_workspace: torch.Tensor | None = None

def get_cublas_workspace_size_bytes() -> int:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
        return 33_554_432
    return 4_194_304


def get_cublas_workspace() -> torch.Tensor:
    """Returns workspace for cublas."""
    global _cublas_workspace
    if _cublas_workspace is None:
        _cublas_workspace = torch.empty(
            get_cublas_workspace_size_bytes(), dtype=torch.uint8, device="cuda"
        )
    return _cublas_workspace

def roundup_div(x: int, y: int) -> int:
    """Round up division"""
    assert x >= 0
    assert y > 0
    return (x + y - 1) // y

def roundup(x: int, y: int) -> int:
    """Round up to multiple of y"""
    assert x >= 0
    assert y > 0
    return (x + y - 1) // y * y

def assert_dim_for_fp4_exec(tensor: torch.Tensor) -> None:
    """Assert that tensor dimensions are supported for FP4 TN GEMM"""
    assert tensor.dim() == 2 and tensor.size(1) % 16 == 0, (
        "FP4 execution requires 2D input matrices with "
        "K dimension divisible by 16 (32 FP4 elements), "
        f"but got tensor with dims={list(tensor.size())}"
    )

def tensor_absmax(x: torch.Tensor) -> torch.Tensor:
    """Compute Tensor amax
    Args:
        x (Tensor): Input tensor.
    Returns:
        Tensor: amax
    """
    absmax = torch.tensor([1.0], dtype=torch.float, device=x.device)
    comfy_quant.ext.compute_tensor_absmax(x, absmax)
    return absmax

def quantize_per_tensor_fp8(x: torch.Tensor, scale: torch.Tensor, output_type: torch.dtype = torch.float8_e4m3fn) -> torch.Tensor:
    """Compute Tensor amax
    Args:
        x (Tensor): Input tensor.
        scale (Tensor): Scale tensor.
        output_type (torch.dtype, optional): FP8 type in [float8_e4m3fn, float8_e5m2] . Defaults to torch.float8_e4m3fn.
    Returns:
        Tensor: Scaled FP8 Tensor
    """
    q_tensor = torch.empty(x.shape, dtype=output_type, device="cuda")
    comfy_quant.ext.compute_quantize_fp8_tensor(x, q_tensor, scale)
    return q_tensor

def quantize_transpose_vector_blockwise_fp4(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    quant_dtype: torch.dtype = torch.float4_e2m1fn_x2,
    block_length: int = 16,
    *,
    return_identity: bool = True,
    return_transpose: bool = True,
    eps: float = 0.0,
    swizzled_scale: bool = True,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Quantizes and optionally transposes x with a (1, block_length)
    quantization tile shape. Each tile of x has a separate scale
    factor computed. qx_transpose is scaled equivalently to
    calculating x_transpose and quantizing it with (1, block_length)
    tiles. In other words, it's scale factors correspond to a tiling
    of non-transposed x by (block_length, 1).

    Parameters:
    x: Tensor of dtype float32 or bf16.
    quant_dtype: only float4_e2m1fn_x2 is supported.
                 use uint8 to store the two quantized values.
    block_length: Quantization tile side length.
        MXFP4: 32
        NVFP4: 16
    return_transpose: Whether to calculate qx_t and sx_t.
    eps: Epsilon used in scale factor calculation.
    pow_2_scale:
        True: Use MXFP4
        False: Use NVFP4
    swizzled_scale: Return swizzled scale for TensorCore if set. Otherwise, return linear scale.
        https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md#scale-factor-layouts
    Returns:
    qx, sx, qx_t, sx_t
    """
    assert x.dim() == 2
    assert x.is_cuda
    assert x.is_contiguous(), "Input tensor must be contiguous"

    assert block_length in [32, 16]
    if block_length == 32:
        scale_dtype = torch.float8_e8m0fnu
        pow_2_scale = True
    else:
        scale_dtype = torch.float8_e4m3fn
        pow_2_scale = False


    assert quant_dtype == torch.float4_e2m1fn_x2, "Only E2M1 is supported."
    transpose_scales = False
    M, N = x.shape

    # allocate output tensors
    if return_identity:
        qx = torch.empty((M, N // 2), device=x.device, dtype=torch.uint8)
        if transpose_scales:
            scale_outer, scale_inner = math.ceil(N / block_length), M
        else:
            if swizzled_scale:
                scale_outer = roundup_div(M, 128) * 128
                scale_inner = (
                    roundup_div(roundup_div(N, block_length), 4) * 4
                )
            else:
                scale_outer, scale_inner = M, roundup_div(N, block_length)

        # for cutlass GEMM, the padding in reduction dimension has to be zero
        # a temporary fix is to initialize the scale to zero if padding is needed
        if swizzled_scale and N % 32 == 0 and N % (block_length * 4) != 0:
            sx = torch.zeros(
                (scale_outer, scale_inner),
                device=x.device,
                dtype=scale_dtype,
            )
        else:
            sx = torch.empty(
                (scale_outer, scale_inner),
                device=x.device,
                dtype=scale_dtype,
            )
    else:
        qx, sx = None, None

    if return_transpose:
        qx_t = torch.empty((N, M // 2), device=x.device, dtype=torch.uint8)
        if transpose_scales:
            scale_outer, scale_inner = math.ceil(M / block_length), N
        else:
            if swizzled_scale:
                scale_outer = roundup_div(N, 128) * 128
                scale_inner = (
                    roundup_div(roundup_div(M, block_length), 4) * 4
                )
            else:
                scale_outer, scale_inner = N, roundup_div(M, block_length)

        # for cutlass GEMM, the padding in reduction dimension has to be zero
        # a temporary fix is to initialize the scale to zero if padding is needed
        if swizzled_scale and M % 32 == 0 and M % (block_length * 4) != 0:
            sx_t = torch.zeros(
                (scale_outer, scale_inner),
                device=x.device,
                dtype=scale_dtype,
            )
        else:
            sx_t = torch.empty(
                (scale_outer, scale_inner),
                device=x.device,
                dtype=scale_dtype,
            )
    else:
        qx_t, sx_t = None, None

    comfy_quant.ext.quantize_transpose_vector_blockwise_fp4(
        x,
        global_amax,
        sx,
        sx_t,
        qx,
        qx_t,
        eps,
        return_identity,
        return_transpose,
        pow_2_scale,
        transpose_scales,
        swizzled_scale,
    )
    return qx, sx, qx_t, sx_t

def quantize_nvfp4(x, amax):
    x, sx, _, _ = quantize_transpose_vector_blockwise_fp4(
            x,
            amax,
            torch.float4_e2m1fn_x2,
            block_length=16,
            return_identity=True,
            return_transpose=False,
            eps = 0.0,
            swizzled_scale = True,
        )
    return x, sx

def fp4_gemm_blockwise(
    a: torch.Tensor,
    a_decode_scale: torch.Tensor,
    b: torch.Tensor,
    b_decode_scale: torch.Tensor,
    block_length: int,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    accumulate: bool = False,
) -> torch.Tensor:
    """cuBLAS FP4 GEMM with block-wise scaling.

    cuBLAS documentation:
    https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-quantization
    It computes the FP4 GEMM operation `y = a * b.T + bias`.

    - A and B are K-major, A: (M, K_a), B: (N, K_b), K_a = K_b
      - A and B use uint8 container to store 2 FP4 elements, FP4 elements in K-dimention is K = 2 * K_a = 2 * K_b
      - K must be divisible by 32
      - N must be divisible by 8

    - Both A and B are FP4 tensors with 1D block-wise quantization
        - MXFP4: 1x32 block (E8M0 scale)
            - Currently not supported for cuBLAS in CUDA 12.9
        - NVFP4: 1x16 block (E4M3 scale)
                 alpha is the global scale factor for the output:
                 alpha = (global_amax_a * global_amax_b) / (6 * 6 * 448 * 448)

    - Scale factor layout requirement
        - A: (RoundUp(M, 128), RoundUp(K//block_length, 4))
        - B: (RoundUp(N, 128), RoundUp(K//block_length, 4))
        - scale factor is in swizzled layout:
            https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md#scale-factor-layouts
            https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout
            cuBLAS and cutlass shares the same tiled scaling factor layout.


    Notes:
        - use_split_accumulator only valid for sm90 FP8 GEMM
    """

    # Check input tensors shapes and dtypes
    assert_dim_for_fp4_exec(a)
    assert_dim_for_fp4_exec(b)
    assert a.dtype == torch.uint8  # 2 FP4 in 1 uint8 container
    assert b.dtype == torch.uint8

    M, K_a = a.shape
    N, K_b = b.shape
    assert K_a == K_b, "Matrix dimensions do not match"

    # K is the number of FP4 elements in a row of a and b
    K = 2 * K_a  # 2 FP4 in 1 uint8 container

    # Check output tensor
    if out is None:
        out = torch.empty(
            M,
            N,
            dtype=out_dtype,
            device=a.device,
        )
    else:
        assert out.shape == (
            M,
            N,
        ), f"Expected shape {(M, N)}, got {out.shape}"
        assert out.is_contiguous(), "Output tensor is not contiguous."
        assert (
            out.dtype == out_dtype
        ), "Output tensor dtype conflicts with the out dtype requested"

    # column of tensor a/b is checked to be divisible by 16 because 2 FP4 in one element
    assert K % 32 == 0, f"Input tensors must have 32 alignment in K dimension"
    assert N % 8 == 0, f"B tensor must have 8 alignment in N dimension"

    # Check scale layout
    assert (
        a_decode_scale.dtype == b_decode_scale.dtype
    ), "A and B scale dtype must match"

    if a_decode_scale.dtype == torch.float8_e8m0fnu:
        # MXFP4: scales are E8M0, and stored in torch.uint8
        assert block_length == 32, "MXFP4 only supports block length 32"
        raise ValueError("MXFP4 is not supported yet for cuBLAS in CUDA 12.9")
    elif a_decode_scale.dtype == torch.float8_e4m3fn:
        # NVFP4: scales are E4M3, and stored in torch.float8_e4m3fn
        assert block_length == 16, "NVFP4 only supports block length 16"
        assert alpha is not None, "alpha must be provided for NVFP4"
        assert alpha.dtype == torch.float32, "alpha must be float32"
        assert alpha.numel() == 1, "alpha must be a scalar"
    else:
        raise ValueError(f"Unsupported scale dtype: {a_decode_scale.dtype}")

    roundup_m = roundup(M, 128)
    roundup_n = roundup(N, 128)
    # K is multiple of 32, so K / block_length is integer,
    roundup_sk = roundup(K // block_length, 4)

    assert a_decode_scale.dim() == 2, "Invalid A scale shape"
    assert a_decode_scale.size() == (roundup_m, roundup_sk), "Invalid A scale shape"

    assert b_decode_scale.dim() == 2, "Invalid B scale shape"
    assert b_decode_scale.size() == (roundup_n, roundup_sk), "Invalid B scale shape"

    if bias is None:
        bias = torch.Tensor()
    else:
        assert bias.dtype in (
            torch.float16,
            torch.bfloat16,
        ), "Only fp16 and bfloat16 bias are supported."

    # NVFP4/MXFP4 in sm100 supports TN layout only
    transa, transb = True, False
    grad = False
    math_sm_count = 0

    comfy_quant.ext.cublas_gemm_blockwise_fp4(
        b,
        b_decode_scale,
        a,
        a_decode_scale,
        out,
        bias,
        torch.Tensor(),  # pre_gelu_out
        transa,
        transb,
        grad,
        get_cublas_workspace(),
        accumulate,
        math_sm_count,
        alpha,
    )

    return out


