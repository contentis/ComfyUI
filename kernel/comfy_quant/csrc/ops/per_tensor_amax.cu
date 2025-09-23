/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "roundup_div.cuh"
#include "utils.cuh"

#include <limits>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace comfy {

namespace {

constexpr int kAbsMaxKernelThreads = 256;

template <int32_t aligned_workunit_size, bool aligned, typename InputType>
__launch_bounds__(kAbsMaxKernelThreads) __global__ void ContiguousRegionAmaxKernel(
    const InputType* input,
    float* amax,
    const uint64_t num_elems,
    const uint32_t num_aligned_workunits) {
  VecLoader<InputType, aligned_workunit_size, aligned> loader(
      input, num_aligned_workunits, num_elems);
  float max{0.0f};
  const int warp_id = threadIdx.x / kThreadsPerWarp;
  const size_t M = num_aligned_workunits;

  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < M; tid += gridDim.x * blockDim.x) {
    loader.LoadOrZero(tid);
#pragma unroll
    for (int i = 0; i < aligned_workunit_size; ++i) {
      const float val = loader.vec.data.ele[i];
      __builtin_assume(max >= 0.0f);
      max = fmaxf(fabsf(val), max);
    }
  }

  max = ReduceMax<kAbsMaxKernelThreads / kThreadsPerWarp>(max, warp_id);
  if (threadIdx.x == 0) {
    AtomicMaxFloat(amax, max);
  }
}

template <int32_t elems_per_aligned_wu, typename InputType>
void ContiguousRegionAmaxLauncher(
    const InputType* input,
    float* amax,
    const int64_t num_elems,
    cudaStream_t stream) {
  // cuda memset amax to zero to allow torch.empty when allocating amax tensor
  cudaMemset(amax, 0, sizeof(float));

  // ContiguousRegionAmaxLauncher runs only if num_elems is not zero,
  // where it always returns a value for amax.
  bool aligned = IsNeatlyAligned(num_elems, elems_per_aligned_wu, input);

  size_t num_aligned_chunks =
      NumAlignedChunksForSpan(input, num_elems, elems_per_aligned_wu, sizeof(InputType));
  size_t num_blocks =
      RoundUpDivide(num_aligned_chunks, static_cast<uint64_t>(kAbsMaxKernelThreads));
  constexpr std::size_t max_blocks = 65535u;

  num_blocks = std::min(num_blocks, max_blocks);

  if (aligned) {
    ContiguousRegionAmaxKernel<elems_per_aligned_wu, true, InputType>
        <<<num_blocks, kAbsMaxKernelThreads, 0, stream>>>(
            input, amax, num_elems, num_aligned_chunks);
  } else {
    ContiguousRegionAmaxKernel<elems_per_aligned_wu, false, InputType>
        <<<num_blocks, kAbsMaxKernelThreads, 0, stream>>>(
            input, amax, num_elems, num_aligned_chunks);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // anonymous namespace

} // namespace comfy

void compute_tensor_absmax(const at::Tensor& input, at::Tensor& amax) {
  // Early exit if input is empty
  if (input.numel() == 0) {
    return;
  }

  // Check tensor sizes
  TORCH_CHECK(amax.dim() == 1, "Amax output must be a 1D tensor.");
  TORCH_CHECK(amax.sizes()[0] == 1, "Whole tensor amax must have a single element.");

  // Check tensor devices, contiguous, and dtypes
  TORCH_CHECK(input.is_cuda(), "input tensor must be on CUDA device.");
  TORCH_CHECK(amax.is_cuda(), "amax tensor must be on CUDA device.");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous.");
  TORCH_CHECK(!comfy::IsFp8Dtype(input.scalar_type()), "Input must not be fp8.");
  // Currently amax has to be in float32. We need to use AtomicMaxFloat for amax,
  // which requires using the integer-based version of atomicMax.
  TORCH_CHECK(amax.scalar_type() == at::kFloat, "Amax must be a float tensor.");

  // Force to make input 1D tensor, assuming input is contiguous
  // If not, view aten op will throw an error
  auto input_flattened = input.view({-1});
  const int64_t num_elems = input_flattened.numel();

  // set up kernel launch parameters
  auto stream = at::cuda::getCurrentCUDAStream();
  // launch kernel
  COMFY_DISPATCH_INPUT_TYPE(
      input.scalar_type(),
      InputType,
      constexpr int32_t elems_per_workunit = comfy::kThreadsPerWarp / sizeof(InputType);
      comfy::ContiguousRegionAmaxLauncher<elems_per_workunit, InputType>(
          reinterpret_cast<const InputType*>(input_flattened.data_ptr()),
          reinterpret_cast<float*>(amax.data_ptr()),
          num_elems,
          stream););
}
