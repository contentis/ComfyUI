/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <cassert>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "utils.cuh"

namespace comfy {

namespace {

void cublas_gemm_blockwise_fp4_impl(
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
    const at::Tensor& alpha,
    cudaStream_t stream) {
  // Sanity checks
  // only TN layout is supported
  TORCH_CHECK(
      transa == true && transb == false, "Only transa == true, transb == false is supported");
  TORCH_CHECK(A.device().is_cuda() && A.is_contiguous(), "A must be CUDA and contiguous");
  TORCH_CHECK(B.device().is_cuda() && B.is_contiguous(), "B must be CUDA and contiguous");
  TORCH_CHECK(D.device().is_cuda() && D.is_contiguous(), "D must be CUDA and contiguous");
  TORCH_CHECK(IsFp4Dtype(A.scalar_type()), "A must be FP4");
  TORCH_CHECK(IsFp4Dtype(B.scalar_type()), "B must be FP4");
  TORCH_CHECK(D.dtype() == at::kBFloat16 || D.dtype() == at::kFloat, "D must be BFloat16 or float");
  TORCH_CHECK(
      A_decode_scale.scalar_type() == at::kFloat8_e4m3fn,
      "A_decode_scale must be FP8 e4m3 for NVFP4");
  TORCH_CHECK(
      B_decode_scale.scalar_type() == at::kFloat8_e4m3fn,
      "B_decode_scale must be FP8 e4m3 for NVFP4");
  TORCH_CHECK(A.dim() == 2, "A must be 2D");
  TORCH_CHECK(B.dim() == 2, "B must be 2D");
  TORCH_CHECK(D.dim() == 2, "D must be 2D");
  // cublas has column major layout, A B has been swapped (eg. A is weight, B is input)
  TORCH_CHECK(D.size(0) == B.size(0) && D.size(1) == A.size(0), "D shape mismatch");
  TORCH_CHECK(alpha.scalar_type() == at::kFloat, "alpha must be float");
  TORCH_CHECK(alpha.numel() == 1, "alpha must be a scalar");

  // m, k, n here are for cuBLAS column major layout
  // this is different with the M N K notation in torch, which is row major layout
  const int m = transa ? A.size(0) : A.size(1);
  // Two fp4 values are packed into one fp8 value, so k is doubled
  const int k = (transa ? A.size(1) : A.size(0)) * 2;
  const int n = transb ? B.size(1) : B.size(0);

  // Handle case where inputs are empty.
  if (m == 0 || n == 0 || k == 0) {
    // For wgrad [n, m] @ [m, k] = [n, k] with m = 0, we need to set D to 0.
    if (D.numel() != 0 && !accumulate) {
      D.fill_(0);
    }
    return;
  }

  int lda = k, ldb = k, ldc = m, ldd = m;
  // D = alpha * (A * B) + beta * C, here alpha is a float32 value
  const float* alpha_ptr = static_cast<const float*>(alpha.data_ptr());

  float* beta_ptr = accumulate ? GetScalarOne() : GetScalarZero();

  cublasLtHandle_t ltHandle;
  TORCH_CUDABLAS_CHECK(cublasLtCreate(&ltHandle));

  // variable to store heuristic result
  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  // Create operation descriptor
  cublasLtMatmulDesc_t operationDesc = nullptr;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

#if CUDA_VERSION >= 12090
  // Setup scaling for A and B
  cublasLtMatmulMatrixScale_t A_scale_mode, B_scale_mode;
  // Note: in cuBLAS term, tensor name A and B are swapped.
  A_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  B_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &A_scale_mode, sizeof(A_scale_mode)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &B_scale_mode, sizeof(B_scale_mode)));
#else
  TORCH_CHECK(false, "NVFP4 cuBLAS GEMM requires CUDA 12.9 or later.");
#endif

  // setup transa and transb (for TN only, transa is true, transb is false)
  // Suppose in fwd pass, A is weight, B is input, D is output
  // transa true: A as weight tensor, with torch shape N x K is a transposed tensor
  // transb false: B as input tensor, with torch shape M x K is a non-transposed tensor
  const cublasOperation_t transa_type = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transb_type = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa_type, sizeof(transa_type)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb_type, sizeof(transb_type)));

  const void* A_decode_scale_ptr = A_decode_scale.data_ptr();
  const void* B_decode_scale_ptr = B_decode_scale.data_ptr();
  // TODO(zhongbo): E2M1 or E0M3?
  // Need to specify that we need E2M1, potentially E0M3 if cublas supports it
  const cudaDataType_t Atype = CUDA_R_4F_E2M1;
  const cudaDataType_t Btype = CUDA_R_4F_E2M1;
  const cudaDataType_t Dtype = at::cuda::ScalarTypeToCudaDataType(D.scalar_type());

  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &A_decode_scale_ptr,
      sizeof(A_decode_scale_ptr)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &B_decode_scale_ptr,
      sizeof(B_decode_scale_ptr)));

  // make sure alpha beta computation dtype remains fp32 by CUBLASLT_MATMUL_DESC_SCALE_TYPE
  cublasDataType_t scale_type = CUDA_R_32F;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

  // Set pointer mode: alpha and beta are both device pointers
  // https://docs.nvidia.com/cuda/cublas/#cublasltpointermode-t
  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));

  // Setup mat layout descriptors
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(
      &Adesc, Atype, transa_type == CUBLAS_OP_N ? m : k, transa_type == CUBLAS_OP_N ? k : m, lda));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(
      &Bdesc, Btype, transb_type == CUBLAS_OP_N ? k : n, transb_type == CUBLAS_OP_N ? n : k, ldb));

  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, Dtype, m, n, ldc));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&Ddesc, Dtype, m, n, ldd));

  // setup epilogue attributes
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  // If bias is provided, add it via cuBLASLt epilogue. Bias is expected to be length m (rows of
  // column-major D, which corresponds to output feature dimension N in row-major view).
  if (bias.numel() != 0) {
    TORCH_CHECK(bias.device().is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(
        (bias.dim() == 1 && bias.size(0) == m) ||
            (bias.dim() == 2 && bias.size(0) == m && bias.size(1) == 1),
        "bias must be shape [N] or [N, 1] where N == A.size(0)");
    TORCH_CHECK(
        bias.scalar_type() == D.scalar_type(),
        "bias dtype must match output dtype");
    epilogue = CUBLASLT_EPILOGUE_BIAS;
    const void* bias_ptr = bias.data_ptr();
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        &bias_ptr,
        sizeof(bias_ptr)));
  }
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

  // setup preference attributes
  cublasLtMatmulPreference_t preference = nullptr;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  size_t workspace_size = workspace.size(0);

  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_size,
      sizeof(workspace_size)));

  // get heuristic result
  const auto status = cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      operationDesc,
      Adesc,
      Bdesc,
      Cdesc,
      Ddesc,
      preference,
      1,
      &heuristicResult,
      &returnedResults);

  TORCH_CHECK(
      status != CUBLAS_STATUS_NOT_SUPPORTED, "Unable to find suitable cuBLAS GEMM algorithm");
  TORCH_CUDABLAS_CHECK(status);
  TORCH_CHECK(returnedResults != 0, "Unable to find any suitable algorithms");

  TORCH_CUDABLAS_CHECK(cublasLtMatmul(
      ltHandle,
      operationDesc,
      alpha_ptr,
      A.data_ptr(),
      Adesc,
      B.data_ptr(),
      Bdesc,
      beta_ptr,
      D.data_ptr(),
      Cdesc,
      D.data_ptr(),
      Ddesc,
      &heuristicResult.algo,
      workspace.data_ptr(),
      workspace_size,
      stream));

  if (preference)
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  if (Ddesc)
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(Ddesc));
  if (Cdesc)
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc)
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc)
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc)
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
//   if (ltHandle)
//     TORCH_CUDABLAS_CHECK(cublasLtDestroy(ltHandle));
}

} // anonymous namespace

} // namespace comfy

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
    const at::Tensor& alpha) {
  comfy::cublas_gemm_blockwise_fp4_impl(
      A,
      A_decode_scale,
      B,
      B_decode_scale,
      D,
      bias,
      pre_gelu_out,
      transa,
      transb,
      grad,
      workspace,
      accumulate,
      math_sm_count,
      alpha,
      at::cuda::getCurrentCUDAStream());
}

// void multi_stream_cublas_gemm_blockwise_fp4() {}
