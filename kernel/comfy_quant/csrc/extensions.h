/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#ifndef COMFY_EXTENSIONS_H_
#define COMFY_EXTENSIONS_H_

#include <cublasLt.h>
#include <cuda.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>


/***************************************************************************************************
 * Quantization
 **************************************************************************************************/

void compute_tensor_absmax(const at::Tensor& input, at::Tensor& amax);

void compute_quantize_fp8_tensor(const at::Tensor &src, at::Tensor &dst, at::Tensor &scale);

void quantize_transpose_vector_blockwise_fp4(
    const at::Tensor& input,
    const at::Tensor& global_amax,
    c10::optional<at::Tensor> scale_inv_opt,
    c10::optional<at::Tensor> scale_inv_t_opt,
    c10::optional<at::Tensor> output_opt,
    c10::optional<at::Tensor> output_t_opt,
    const float epsilon,
    const bool return_identity,
    const bool return_transpose,
    const bool pow2_scale,
    const bool transpose_scales,
    const bool swizzled_scale);

void cublas_gemm_blockwise_fp4(
    const at::Tensor& A,
    const at::Tensor& A_decode_scale,
    const at::Tensor& B,
    const at::Tensor& B_decode_scale,
    at::Tensor& D,
    const at::Tensor& bias,
    at::Tensor& pre_gelu_out,
    bool transa,
    bool transb,
    bool grad,
    at::Tensor& workspace,
    bool accumulate,
    int math_sm_count,
    const at::Tensor& alpha);

namespace comfy {

constexpr int kNumGEMMStreams = 4;

} // namespace comfy

#endif // COMFY_EXTENSIONS_H_
