/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "roundup_div.cuh"
#include "utils.cuh"
#include "device_math.cuh"
#include <limits>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

constexpr int kQMaxKernelThreads = 128;
constexpr int kE4M3Alignment = 16;

namespace comfy {

namespace {

template<typename InputType>
__forceinline__ __device__ float convert_to_float(InputType val) {
        float src_val_float;
        if constexpr (std::is_same_v<InputType, float>) {
            src_val_float = val;
        } else if constexpr (std::is_same_v<InputType, __half>) {
            src_val_float = __half2float(val);
        } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
            src_val_float = __bfloat162float(val);
        }
        return src_val_float;
}

__forceinline__ __device__ float clamp(float val, float min, float max) {
	return fminf(max, fmaxf(min, val));
}


template<typename InputType, typename OutputType>
__global__ void quantize_fp8_tensor_kernel(const InputType *src, OutputType *dst, const float *scale_f, const uint32_t size)
{
    constexpr float kFP8Max = FP8LimitsTrait<OutputType>::max;
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	idx *= kE4M3Alignment;
	if (idx >= size) {
		return;
	}
	constexpr int n_src_load = sizeof(float4) / sizeof(InputType); // 8
	constexpr int n_loads = kE4M3Alignment / n_src_load; // 2

    union {
        float4 f4;
        OutputType f8[kE4M3Alignment];
    } f4_e4m3;

#pragma unroll
	for(int i = 0; i < n_loads; i++) {
		float4 _src_f4 = *reinterpret_cast<const float4*>(src + idx + i * n_src_load);
		InputType *_src_ptr = reinterpret_cast<InputType*>(&_src_f4);
#pragma unroll
		for(int j = 0; j < n_src_load; j++) {
			float scaled_val = convert_to_float(_src_ptr[j]) / *scale_f;
			scaled_val = clamp(scaled_val, -kFP8Max, kFP8Max);
			f4_e4m3.f8[i*n_src_load + j] = static_cast<OutputType>(scaled_val);
		}
	}
	*reinterpret_cast<float4*>(dst + idx) = f4_e4m3.f4;
}

} // anonymous namespace
} // namespace comfy

void compute_quantize_fp8_tensor(const at::Tensor &src, at::Tensor &dst, at::Tensor &scale)
{
  // Check tensor sizes
  TORCH_CHECK(scale.numel() ==1, "scale needs to be a scalar");

  // Check tensor devices, contiguous, and dtypes
  TORCH_CHECK(src.is_cuda(), "input tensor must be on CUDA device.");
  TORCH_CHECK(dst.is_cuda(), "output tensor must be on CUDA device.");
  TORCH_CHECK(scale.is_cuda(), "amax tensor must be on CUDA device.");

  TORCH_CHECK(src.is_contiguous(), "Input must be contiguous.");
  TORCH_CHECK(dst.is_contiguous(), "Output must be contiguous.");
  TORCH_CHECK(!comfy::IsFp8Dtype(src.scalar_type()), "Input must not be fp8.");
  TORCH_CHECK(comfy::IsFp8Dtype(dst.scalar_type()), "Output must be fp8.");
  TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be a float tensor.");
  TORCH_CHECK(src.numel() == dst.numel(), "input and output must have the same size");

  auto src_flattened = src.view({-1});
  auto dst_flattened = dst.view({-1});
  const int64_t num_elems = src_flattened.numel();
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int vals_per_thread = kE4M3Alignment;
  constexpr int vals_per_block = vals_per_thread * kQMaxKernelThreads;
  const int blocks = static_cast<int>((num_elems + vals_per_block - 1) / vals_per_block);

  COMFY_DISPATCH_INPUT_TYPE(
      src.scalar_type(),
      InputType,
      COMFY_DISPATCH_FP8_TYPE(
        dst.scalar_type(),
        OutputType,
        comfy::quantize_fp8_tensor_kernel<InputType, OutputType><<<blocks, kQMaxKernelThreads, 0, stream>>>(
        reinterpret_cast<const InputType*>(src_flattened.data_ptr()),
        reinterpret_cast<OutputType*>(dst_flattened.data_ptr()),
        reinterpret_cast<float*>(scale.data_ptr()),
        num_elems);
      ) // OutputType
  ) // InputType
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

