/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#ifndef COMFY_FLOAT32_TO_E8M0_UTILS_CUH_
#define COMFY_FLOAT32_TO_E8M0_UTILS_CUH_

#include <cuda.h>
#include <cuda_fp8.h>
#include <cmath>

namespace comfy {

// Reference implementation ported from TransformerEngine
// https://github.com/NVIDIA/TransformerEngine/blob/097afc00d72800ca7328ae1ff8a0d84399b51880/transformer_engine/common/utils.cuh#L933
__device__ __forceinline__ uint8_t float_to_e8m0_ru_helper(float val) {
#if CUDA_VERSION >= 12080
  constexpr cudaRoundMode kRoundingMode = cudaRoundPosInf;
  constexpr __nv_saturation_t kSaturation = __NV_SATFINITE;
  __nv_fp8_storage_t biased_exponent = __nv_cvt_float_to_e8m0(val, kSaturation, kRoundingMode);
  return static_cast<uint8_t>(biased_exponent);
#else
  TORCH_CHECK(false, "float_to_e8m0_ru conversion requires CUDA 12.8+");
  return 0xFF;
#endif
}

// nan -> nan
// inf -> 2 ** -127
// 0 -> 1.0
// Others: 2 ** (127 - biased_exp)
__device__ __forceinline__ float exp2f_rcp(uint8_t biased_exp) {
#define FP32_MANTISSA_BITS 23
  uint32_t result = static_cast<uint32_t>(uint8_t(254) - biased_exp)
      << FP32_MANTISSA_BITS; // Division
  result = biased_exp == 0 ? 0x3f800000 : result; // 1.0
  result = biased_exp == 254 ? 0x00400000 : result; // denormal number
  // https://en.wikipedia.org/wiki/Single-precision_floating-point_format
  result += (biased_exp == 255); // nan
  return __uint_as_float(result);
#undef FP32_MANTISSA_BITS
}

} // namespace comfy

#endif // COMFY_FLOAT32_TO_E8M0_UTILS_CUH_
