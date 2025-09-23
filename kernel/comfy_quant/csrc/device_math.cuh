/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#ifndef COMFY_DEVICE_MATH_CUH_
#define COMFY_DEVICE_MATH_CUH_

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <limits>

#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif

#include "float_to_e8m0_utils.cuh"

namespace comfy {

using std::int32_t;
using std::uint32_t;
using std::uint8_t;

#if CUDA_VERSION >= 12080
template <typename T>
struct FP4LimitsTrait;

template <>
struct FP4LimitsTrait<__nv_fp4x2_storage_t> {
  static constexpr float max = 6.0f;
  static constexpr float max_inverse = 1.0 / max;
};
#endif
// Type trait for extreme values of fp8 types.
// Used in the calculation of scale factors
// as a constexpr lookup from e4m3 or e5m2 to
// the max finite value.
template <typename T>
struct FP8LimitsTrait;

template <>
struct FP8LimitsTrait<__nv_fp8_e4m3> {
  static constexpr float max = 448.0f;
  static constexpr float max_inverse = 1.0 / max;
};

template <>
struct FP8LimitsTrait<__nv_fp8_e5m2> {
  static constexpr float max = 57344.0f;
  static constexpr float max_inverse = 1.0 / max;
};

// Type trait to resolve the max finite value
// represented by a input type to quantization.
// Or to represent max representable power of 2
// finite value.
template <typename T, bool ForcePow2>
struct HighPrecisionFloatScaleLimitsTrait;

template <>
struct HighPrecisionFloatScaleLimitsTrait<float, false> {
  static constexpr float max = std::numeric_limits<float>::max();
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<float, true> {
  // Hex float format of 1.0 * 2 ^ 127
  static constexpr float max = 0x1.0p127;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<nv_bfloat16, false> {
  // Hex float format of 1.(7 bits of 1) * 2 ^ 127
  static constexpr float max = 0x1.FEp127;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<nv_bfloat16, true> {
  // Hex float format of 1.0 * 2 ^ 127
  static constexpr float max = 0x1.0p127;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<half, false> {
  // Hex float format of 1.(10 bits of 1) * 2 ^ 15
  static constexpr float max = 0x1.FFCp15;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<half, true> {
  // Hex float format of 1.0 * 2 ^ 15
  static constexpr float max = 0x1.0p15;
};

// Calculate the quantization scale for an individual data element
// given the amax(abs(tile)) value for a given quantization tile.
//
//
// Arguments:
// IType: data type of the tensor being quantized (float or bf16)
// OType: quantized data type (e4m3 or e5m2)
// pow_2_scaling: Whether to force the scale to be a power of 2.
// amax: The evaluation of amax(abs(tile)) for the quantization tile.
// eps: An epsilon used as a floor for amax.
template <typename IType, typename OType, bool Power2Scaling>
__device__ __forceinline__ float ComputeScale(const float amax, const float eps) {
  constexpr float kFP8Max = FP8LimitsTrait<OType>::max;

  // Clamping amax to avoid division by small numbers
  float amax_mod = fmaxf(amax, eps);

  // Handle overflow cases for non-clamped amax (eps is 0 or very small)
  if (amax_mod == 0.f) {
    // If amax is 0, return 1
    return 1.f;
  }
  // Compute scale factor
  float scale = kFP8Max / amax_mod;

  if (isinf(scale)) {
    // If scale is infinity, return max value of IType
    return HighPrecisionFloatScaleLimitsTrait<IType, Power2Scaling>::max;
  }
  if (scale == 0.0) {
    // Case that amax is "inf". The frexp, ldexp logic changes 0.0 scales.
    // Return 0.0 for 0.0 scale here is consistent with non-Power2Scaling model.
    // quantization will remove signal from the tensor,
    // this is bad for the model, but define pow2Scale behavior
    // as returning 0.0 scale. amax calculation can
    // improve the situation to avoid this by taking largest finite.
    return scale;
  }
  if constexpr (Power2Scaling) {
    // NOTE: using bit fiddling based on advice of Asit in this
    // thread: https://nvidia.slack.com/archives/C06EDT7LZEW/p1738274404153439

    uint32_t scale_bits = *reinterpret_cast<uint32_t*>(&scale);
    // Scale must be positive, shift it
    uint8_t exp = scale_bits >> 23;

    // inf scales already early returned, as did nan scales.
    // The cases to consider here are normals, zero, and subnormals.
    // zero is not possible with current math as
    // 448.0 / float_max == 1.31655e-36, which is the smallest
    // possible scale given current dtypes. It is still in the normal
    // fp32 range with an exponent of -120, so subnormals are also
    // not possible.

    int32_t normal_biased_exp = static_cast<int32_t>(exp) - 127;
    __builtin_assume(exp != 0);
    // Normal numbers case.

    // TODO: When combining with URM, can be useful to avoid exponent==127 edge
    // cases where a random mantissa corresponds to a floating point special.
    // Consider how to avoid those pitfalls if/when adding URM.
    scale = ldexpf(1.0f, normal_biased_exp);
  }
  return scale;
}

#if CUDA_VERSION >= 12080
// Calculate the quantization scale for an individual data element
// given the amax(abs(tile)) value for a given quantization tile.
//
// Arguments:
// scale_type/pow_2_scaling: E8M0 or E4M3 scale
// amax: The evaluation of amax(abs(tile)) for the quantization tile.
// global_scale: The global scale factor, only used for NVFP4 (Power2Scaling=false).

template <typename OType, typename ScaleType, bool Power2Scaling>
__device__ __forceinline__ ScaleType
ComputeDecodeScaleFP4(const float amax, const float global_encode_scale) {
  // Compute decode scale factor
  if constexpr (Power2Scaling) {
    float decode_scale = amax * FP4LimitsTrait<OType>::max_inverse;
    if (isinf(decode_scale)) {
      return static_cast<ScaleType>(HighPrecisionFloatScaleLimitsTrait<float, true>::max);
    }
    auto out = float_to_e8m0_ru_helper(decode_scale);
    return *reinterpret_cast<ScaleType*>(&out);
  } else {
    float decode_scale = amax / FP4LimitsTrait<OType>::max;
    decode_scale = decode_scale * global_encode_scale;
    decode_scale = fminf(decode_scale, HighPrecisionFloatScaleLimitsTrait<float, false>::max);
    return static_cast<ScaleType>(decode_scale);
  }
}

// Calculate the encode scale factor for a given decode scale factor.
//
// Arguments:
// decode_scale: The decode scale factor for the quantization.
// global_encode_scale: The global encode scale factor.
// if Power2Scaling is false (E4M3), the output needs to multiply the global encode scale.
template <typename ScaleType, bool Power2Scaling>
__device__ __forceinline__ float ComputeEncodeScaleFP4(
    ScaleType decode_scale,
    const float global_decode_scale) {
  if constexpr (Power2Scaling) {
    // for E8M0, the smallest value of decode_scale is 0x1.0p-127f
    return 1.0f / static_cast<float>(decode_scale);
  } else {
    // for E4M3, the smallest value is 0.f, avoid overflow of encode scale
    // NOTE: This is written in a weird way to match with the implementation of
    // psx-formats.
    return fminf(
        1.0f / (static_cast<float>(decode_scale) * global_decode_scale),
        HighPrecisionFloatScaleLimitsTrait<float, false>::max);
  }
}

// Calculate the output value for a given input value and scale factor.
//
// Arguments:
// input: The input value to be quantized.
// encode_scale: The encode scale factor for the quantization.
template <typename IType, typename ScaleType, bool Power2Scaling>
__device__ __forceinline__ float ComputeOutputFP4(IType input, float encode_scale) {
  return static_cast<float>(input) * encode_scale;
}

// calculate the global encode scale factor for a given global amax.
__device__ __forceinline__ float ComputeGlobalEncodeScaleFP4(const float global_amax) {
  constexpr float fp8_max = FP8LimitsTrait<__nv_fp8_e4m3>::max;
  constexpr float fp4_max = FP4LimitsTrait<__nv_fp4x2_storage_t>::max;
  float global_encode_scale = fp8_max * fp4_max / global_amax;
  // If scale is infinity, return max value of float32
  global_encode_scale =
      fminf(global_encode_scale, HighPrecisionFloatScaleLimitsTrait<float, false>::max);
  // If global amax is 0 or infinity, return 1
  if (global_amax == 0.f || global_encode_scale == 0.f) {
    return 1.f;
  }
  return global_encode_scale;
}

#endif // if CUDA_VERSION >= 12080

} // namespace comfy

#endif // ifndef COMFY_DEVICE_MATH_CUH_
