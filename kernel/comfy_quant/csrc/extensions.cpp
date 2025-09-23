/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <torch/extension.h>

#include "extensions.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Quantization functions
    m.def(
      "compute_tensor_absmax",
      &compute_tensor_absmax,
      "Compute absolute maximum of an entire tensor.");
    m.def(
      "compute_quantize_fp8_tensor",
      &compute_quantize_fp8_tensor,
      "Compute FP8 quantized tensor using a single scale.");
    m.def(
      "quantize_transpose_vector_blockwise_fp4",
      &quantize_transpose_vector_blockwise_fp4,
      "Fused FP4 quantize with block-wise quantization with 1D blocks. "
      "E.g. 1x32 quantization blocks.");
    m.def(
      "cublas_gemm_blockwise_fp4",
      &cublas_gemm_blockwise_fp4,
      "Compute GEMM with cuBLASLt blockwise with FP4 quantization");
 }
