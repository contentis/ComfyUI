/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#ifndef COMFY_UTILS_CUH_
#define COMFY_UTILS_CUH_

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif

#include <type_traits>

#include <c10/cuda/CUDAFunctions.h>
#include <torch/extension.h>
#include <mutex>

#include "extensions.h"
#include "roundup_div.cuh"
namespace comfy {

////////////////////////////////////////////////////////////////////////////////

constexpr int kThreadsPerWarp = 32;

////////////////////////////////////////////////////////////////////////////////

template <int kNumElems>
__device__ __forceinline__ float WarpReduceMax(const float per_thread_max) {
  // kNumElems must be a power of 2 and <= 32
  static_assert(kNumElems <= kThreadsPerWarp, "kNumElems must be <= kThreadsPerWarp (32)");
  static_assert((kNumElems & (kNumElems - 1)) == 0, "kNumElems must be a power of 2");
  // reduction using warp shuffling
  float current_max = per_thread_max;
#pragma unroll
  for (int delta = kNumElems / 2; delta > 0; delta /= 2) {
    const float received_max = __shfl_down_sync(0xFFFFFFFF, current_max, delta);
    __builtin_assume(current_max >= 0.0f);
    __builtin_assume(received_max >= 0.0f);
    current_max = fmaxf(current_max, received_max);
  }
  return current_max;
}

template <int kN>
__device__ __host__ consteval int NextPowerOf2() {
  static_assert(kN > 0, "kN must be > 0");
  // Round up to the next power of 2 by counting leading zeros.
  return 1 << (32 - __builtin_clz(kN - 1));
}

template <int kNumWarps, typename CType>
__device__ __forceinline__ CType ReduceMax(const CType per_thread_max, const int warpid) {
  // intra-warp reduction
  constexpr int kWarpSize = 32;
  const float my_max = per_thread_max;
  const float my_warp_max = WarpReduceMax<kWarpSize>(my_max);

  // inter-warp reduction
  __shared__ float staging[kNumWarps];
  if (threadIdx.x % 32 == 0) {
    staging[warpid] = my_warp_max;
  }
  __syncthreads();
  CType result = 0;
  if (warpid == 0) {
    const float my_max = threadIdx.x < kNumWarps ? staging[threadIdx.x] : 0.0f;
    constexpr int kNumWarpsPow2 = NextPowerOf2<kNumWarps>();
    result = WarpReduceMax<kNumWarpsPow2>(my_max);
  }
  return result;
}

// Works only on positive values
__device__ __forceinline__ void AtomicMaxFloat(float* addr, const float value) {
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(value));
}

////////////////////////////////////////////////////////////////////////////////

// Type trait that maps numeric types in CUDA to their equivalent PyTorch types.
template <typename T>
struct TorchValueType;

template <>
struct TorchValueType<float> {
  using type = float;
};

template <>
struct TorchValueType<half> {
  using type = at::Half;
};

template <>
struct TorchValueType<nv_bfloat16> {
  using type = at::BFloat16;
};

template <>
struct TorchValueType<__nv_fp8_e4m3> {
  using type = at::Float8_e4m3fn;
};

template <>
struct TorchValueType<__nv_fp8_e5m2> {
  using type = at::Float8_e5m2;
};

////////////////////////////////////////////////////////////////////////////////

#define COMFY_DISPATCH_INPUT_TYPE(dtype, type, ...) \
  switch (dtype) {                                    \
    case at::ScalarType::Float: {                     \
      using type = float;                             \
      {                                               \
        __VA_ARGS__                                   \
      }                                               \
    } break;                                          \
    case at::ScalarType::Half: {                      \
      using type = half;                              \
      {                                               \
        __VA_ARGS__                                   \
      }                                               \
    } break;                                          \
    case at::ScalarType::BFloat16: {                  \
      using type = nv_bfloat16;                       \
      {                                               \
        __VA_ARGS__                                   \
      }                                               \
    } break;                                          \
    default:                                          \
      TORCH_CHECK(false, "Unsupported input type.");  \
  }

#define COMFY_DISPATCH_FP8_TYPE(dtype, type, ...) \
  switch (dtype) {                                  \
    case at::ScalarType::Float8_e4m3fn: {           \
      using type = __nv_fp8_e4m3;                   \
      {                                             \
        __VA_ARGS__                                 \
      }                                             \
    } break;                                        \
    case at::ScalarType::Float8_e5m2: {             \
      using type = __nv_fp8_e5m2;                   \
      {                                             \
        __VA_ARGS__                                 \
      }                                             \
    } break;                                        \
    default:                                        \
      TORCH_CHECK(false, "Unsupported FP8 type.");  \
  }

#if CUDA_VERSION >= 12080
#define COMFY_DISPATCH_FP4_TYPE(dtype, type, ...) \
  switch (dtype) {                                  \
    case at::ScalarType::Byte: {                    \
      using type = __nv_fp4x2_storage_t;            \
      {                                             \
        __VA_ARGS__                                 \
      }                                             \
    } break;                                        \
    default:                                        \
      TORCH_CHECK(false, "Unsupported FP4 type.");  \
  }

#define COMFY_DISPATCH_SCALE_TYPE_FP4(pow2_scale, ScaleType, kScaleBlockDim, ...) \
  {                                                                                 \
    if (pow2_scale) {                                                               \
      using ScaleType = __nv_fp8_e8m0;                                              \
      constexpr int kScaleBlockDim = 32;                                            \
      {                                                                             \
        __VA_ARGS__                                                                 \
      }                                                                             \
    } else {                                                                        \
      using ScaleType = __nv_fp8_e4m3;                                              \
      constexpr int kScaleBlockDim = 16;                                            \
      {                                                                             \
        __VA_ARGS__                                                                 \
      }                                                                             \
    }                                                                               \
  }

#endif // if CUDA_VERSION >= 12080

#define COMFY_DISPATCH_INDEX_TYPE(int_overflow, IndexType, ...) \
  {                                                               \
    if (int_overflow) {                                           \
      using IndexType = int64_t;                                  \
      {                                                           \
        __VA_ARGS__                                               \
      }                                                           \
    } else {                                                      \
      using IndexType = int;                                      \
      {                                                           \
        __VA_ARGS__                                               \
      }                                                           \
    }                                                             \
  }

#define COMFY_DISPATCH_BOOL(condition, ConstName, ...) \
  {                                                      \
    if (condition) {                                     \
      constexpr bool ConstName = true;                   \
      {                                                  \
        __VA_ARGS__                                      \
      }                                                  \
    } else {                                             \
      constexpr bool ConstName = false;                  \
      {                                                  \
        __VA_ARGS__                                      \
      }                                                  \
    }                                                    \
  }

////////////////////////////////////////////////////////////////////////////////
#if CUDA_VERSION >= 12080
inline bool IsFp4Dtype(const at::ScalarType dtype) {
  // FP4 values are currently stored in uint8 (Byte) containers
  return dtype == at::ScalarType::Byte;
}
#endif

inline bool IsFp8Dtype(const at::ScalarType dtype) {
  return dtype == at::ScalarType::Float8_e4m3fn || dtype == at::ScalarType::Float8_e5m2;
}

constexpr inline uint64_t CalcAlignment(const void* ptr, const uint64_t size) {
  uint64_t ptr_as_number = reinterpret_cast<uint64_t>(ptr);
  return ptr_as_number % size;
}

template <typename T>
constexpr bool IsNeatlyAligned(
    uint64_t collection_size,
    int32_t aligned_element_count,
    const T* collection_begin) {
  return (
      CalcAlignment(collection_begin, sizeof(T) * static_cast<uint64_t>(aligned_element_count)) ==
          0 &&
      collection_size % aligned_element_count == 0);
}

inline uint64_t NumAlignedChunksForSpan(
    const void* ptr,
    uint64_t num_elems,
    int32_t element_align_count,
    uint64_t element_itemsize) {
  uint64_t alignment_bytes =
      CalcAlignment(ptr, static_cast<uint64_t>(element_align_count) * element_itemsize);
  uint64_t aligment_elems = alignment_bytes / element_itemsize;
  return RoundUpDivide(num_elems + aligment_elems, static_cast<uint64_t>(element_align_count));
}

////////////////////////////////////////////////////////////////////////////////

// Utilities for vectorized loads and stores
template <int kBytes>
struct BytesToType {};

template <>
struct BytesToType<1> {
  using type = uint8_t;
  static_assert(sizeof(type) == 1);
};

template <>
struct BytesToType<2> {
  using type = uint16_t;
  static_assert(sizeof(type) == 2);
};

template <>
struct BytesToType<4> {
  using type = uint32_t;
  static_assert(sizeof(type) == 4);
};

template <>
struct BytesToType<8> {
  using type = uint64_t;
  static_assert(sizeof(type) == 8);
};

template <>
struct BytesToType<16> {
  using type = uint4;
  static_assert(sizeof(type) == 16);
};

struct uint8 {
  uint4 u1, u2;
};

struct uint16 {
  uint4 u1, u2, u3, u4;
};

template <>
struct BytesToType<32> {
  using type = uint8;
  static_assert(sizeof(type) == 32);
};

template <>
struct BytesToType<64> {
  using type = uint16;
  static_assert(sizeof(type) == 64);
};

// Struct for vectorized loads and stores
template <typename EleType, uint32_t kNumEle>
struct Vec {
  static constexpr int kBytes = kNumEle * sizeof(EleType);
  using VecType = typename BytesToType<kBytes>::type;

  // Union for vector or element-wise data access
  using DataType = union {
    VecType vec;
    EleType ele[kNumEle];
  };

  DataType data;

  // Vectorized load data from memory, interpreting the pointer as VecType.
  inline __device__ void VecLoadFrom(const void* base_ptr, int64_t idx = 0) {
    this->data.vec = static_cast<const VecType*>(base_ptr)[idx];
  }

  // Vectorized store data to memory, interpreting the pointer as VecType.
  inline __device__ void VecStoreTo(void* base_ptr, int64_t idx = 0) const {
    static_cast<VecType*>(base_ptr)[idx] = this->data.vec;
  }

  // If the pointer is unaligned or `num_ele` is less than `kNumEle`, load data element-wise from
  // memory. The remaining elements are set to zero.
  inline __device__ void EleLoadFromIfNeeded(
      const void* base_ptr,
      int64_t idx = 0,
      int num_ele = kNumEle) {
    const EleType* ele_ptr = static_cast<const EleType*>(base_ptr) + idx;
    bool is_unaligned = reinterpret_cast<uintptr_t>(ele_ptr) % kBytes != 0;
    // element-wise load
    if (is_unaligned || num_ele < kNumEle) {
#pragma unroll
      for (int i = 0; i < kNumEle; i++) {
        EleType val = (i < num_ele ? ele_ptr[i] : static_cast<EleType>(0.f));
        this->data.ele[i] = val;
      }
    } else {
      // vectorized load
      this->VecLoadFrom(ele_ptr);
    }
  }

  // If the pointer is unaligned or `num_ele` is less than `kNumEle`, store data element-wise to
  // memory.
  inline __device__ void EleStoreToIfNeeded(void* base_ptr, int64_t idx = 0, int num_ele = kNumEle)
      const {
    EleType* ele_ptr = static_cast<EleType*>(base_ptr) + idx;
    bool is_unaligned = reinterpret_cast<uintptr_t>(ele_ptr) % kBytes != 0;
    // element-wise store
    if (is_unaligned || num_ele < kNumEle) {
#pragma unroll
      for (int i = 0; i < kNumEle; i++) {
        if (i < num_ele) {
          ele_ptr[i] = this->data.ele[i];
        }
      }
    } else {
      // vectorized store
      this->VecStoreTo(ele_ptr);
    }
  }

  // Set all elements to zero
  inline __device__ void clear() {
#pragma unroll
    for (int i = 0; i < kNumEle; i++) {
      this->data.ele[i] = static_cast<EleType>(0.f);
    }
  }
};

template <typename EleType, uint32_t kVecSize, bool aligned>
class VecLoader {
 public:
  __device__ VecLoader(const EleType* base_ptr, uint64_t total_vecs, uint64_t total_elems)
      : vec(), base_ptr_(base_ptr), total_vecs_(total_vecs), total_elems_(total_elems) {}

  __device__ void LoadOrZero(int64_t vec_idx) {
    if constexpr (aligned) {
      if (vec_idx < total_vecs_) {
        vec.VecLoadFrom(base_ptr_, vec_idx);
      } else {
        vec.clear();
      }
    } else {
      if (vec_idx < total_vecs_ - 1) {
        vec.EleLoadFromIfNeeded(base_ptr_, vec_idx * kElesPerVec);
      } else if (vec_idx == total_vecs_ - 1) {
        vec.EleLoadFromIfNeeded(
            base_ptr_,
            vec_idx * kElesPerVec,
            kElesPerVec - (total_vecs_ * kElesPerVec - total_elems_));
      } else {
        vec.clear();
      }
    }
  }

  Vec<EleType, kVecSize> vec;

 private:
  static constexpr uint64_t kElesPerVec = kVecSize;
  const EleType* base_ptr_;
  uint64_t total_vecs_;
  uint64_t total_elems_;
};

/* Use CUDA const memory to store scalar 1 and 0 for cublas usage
 */
__device__ __constant__ float one_device;
__device__ __constant__ float zero_device;

inline float* GetScalarOne() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    float one = 1.0f;
    C10_CUDA_CHECK(cudaMemcpyToSymbol(one_device, &one, sizeof(float)));
  });
  // return address by cudaGetSymbolAddress
  float* dev_ptr;
  C10_CUDA_CHECK(cudaGetSymbolAddress((void**)&dev_ptr, one_device));
  return dev_ptr;
}

inline float* GetScalarZero() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    float zero = 0.0f;
    C10_CUDA_CHECK(cudaMemcpyToSymbol(zero_device, &zero, sizeof(float)));
  });
  // return address by cudaGetSymbolAddress
  float* dev_ptr;
  C10_CUDA_CHECK(cudaGetSymbolAddress((void**)&dev_ptr, zero_device));
  return dev_ptr;
}

/* The streams are persistent to reuse across the program execution,
 * but they are local to each cpp translation unit. This should be fine
 * because in PyTorch distributed training, we follow the one GPU per process strategy.
 */
inline cudaStream_t GetGEMMStream(int index) {
  static cudaStream_t streams[kNumGEMMStreams];
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    for (int i = 0; i < kNumGEMMStreams; i++) {
      C10_CUDA_CHECK(cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, 0));
    }
  });
  return streams[index];
}

/* The events are persistent to reuse across the program execution,
 * but they are local to each cpp translation unit. This should be fine
 * because in PyTorch distributed training, we follow the one GPU per process strategy.
 */
inline cudaEvent_t GetCUDAEvent(int index) {
  static cudaEvent_t events[kNumGEMMStreams];
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    for (int i = 0; i < kNumGEMMStreams; i++) {
      C10_CUDA_CHECK(cudaEventCreate(&events[i]));
    }
  });
  return events[index];
}

} // namespace comfy

#endif // COMFY_UTILS_CUH_
