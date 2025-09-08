#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <ATen/ATen.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float4_e2m1fn_x2.h>
#include <cstdint>
#include <cmath>
#include <cstddef>

#define E4M3_ALIGNMENT 16
#define NVFP4_BLOCK_SIZE 16

#define E4M3_AMAX 448.0f
#define E4M3_EPS 0.001953f
#define E2M1_AMAX 6.0f

__forceinline__ __device__ float convert_to_float(__half val) {
	return __half2float(val);
}
__forceinline__ __device__ float convert_to_float(__nv_bfloat16 val) {
	return __bfloat162float(val);
}
__forceinline__ __device__ float clamp(float val, float min, float max) {
	return fminf(max, fmaxf(min, val));
}


template<typename T>
__global__ void quantize_e4m3_tensor_kernel(const T *src, __nv_fp8_e4m3 *dst, const float *scale_f, const uint32_t size)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	idx *= E4M3_ALIGNMENT;
	if (idx >= size) {
		return;
	}
	constexpr int n_src_load = sizeof(float4) / sizeof(T); // 8
	constexpr int n_loads = E4M3_ALIGNMENT / n_src_load; // 2

    union {
        float4 f4;
        __nv_fp8_e4m3 f8[E4M3_ALIGNMENT];
    } f4_e4m3;

#pragma unroll
	for(int i = 0; i < n_loads; i++) {
		float4 _src_f4 = *reinterpret_cast<const float4*>(src + idx + i * n_src_load);
		T *_src_ptr = reinterpret_cast<T*>(&_src_f4);
#pragma unroll
		for(int j = 0; j < n_src_load; j++) {
			float scaled_val = convert_to_float(_src_ptr[j]) / *scale_f;
			scaled_val = clamp(scaled_val, -E4M3_AMAX, E4M3_AMAX);
			f4_e4m3.f8[i*n_src_load + j] = static_cast<__nv_fp8_e4m3>(scaled_val);
		}
	}
	*reinterpret_cast<float4*>(dst + idx) = f4_e4m3.f4;
}


template<typename T>
__global__ void quantize_nvfp4_kernel(const T *src, __nv_fp4x2_e2m1 *dst, const float *ss_f32, __nv_fp8_e4m3 *block_scale_out, const uint32_t size)
{
	uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t idx = thread_id * NVFP4_BLOCK_SIZE;
	if (idx >= size) {
		return;
	}

	constexpr int n_src_load = sizeof(float4) / sizeof(T); // 8
	constexpr int n_loads = NVFP4_BLOCK_SIZE / n_src_load; // 2

    union {
        float2 f2;
        __nv_fp4x2_e2m1 f4x2[NVFP4_BLOCK_SIZE / 2];
    } f2_e2m1;
    // T amax_hp = static_cast<T>(0.0f);
    T amax_hp;

    // block scale
#pragma unroll
	for(int i = 0; i < n_loads; i++) {
		float4 _src_f4 = *reinterpret_cast<const float4*>(src + idx + i * n_src_load);
		T *_src_ptr = reinterpret_cast<T*>(&_src_f4);
#pragma unroll
		for(int j = 0; j < n_src_load; j++) {
			amax_hp = __hmax(amax_hp, __habs(_src_ptr[j]));
		}

	}
	float s_hp = convert_to_float(amax_hp) / E2M1_AMAX;

    // quantize(s_hp, ss_f32)
	float total_scale = clamp(s_hp / *ss_f32, E4M3_EPS, E4M3_AMAX);
	__nv_fp8_e4m3 s_e4m3 = static_cast<__nv_fp8_e4m3>(total_scale);

    // quantize(x_hp, s_e4m3)
    float2 scaled_val;
	float s_e4m3_f32 = static_cast<float>(s_e4m3);
#pragma unroll
	for(int i = 0; i < NVFP4_BLOCK_SIZE / 2; i++) {
	    scaled_val.x = convert_to_float(src[idx + i*2]);
	    scaled_val.y = convert_to_float(src[idx + i*2 + 1]);
	    scaled_val.x = clamp(scaled_val.x / s_e4m3_f32, -E2M1_AMAX, E2M1_AMAX);
	    scaled_val.y = clamp(scaled_val.y / s_e4m3_f32, -E2M1_AMAX, E2M1_AMAX);
		f2_e2m1.f4x2[i] = static_cast<__nv_fp4x2_e2m1>(scaled_val);
	}

	*reinterpret_cast<float2*>(dst + idx) = f2_e2m1.f2;
	block_scale_out[thread_id] = s_e4m3;
}

void launch_quantize_e4m3_tensor(const at::Tensor &src, at::Tensor &dst, at::Tensor &scale, cudaStream_t stream)
{
	const uint64_t numel = static_cast<uint64_t>(src.numel());
	const int vals_per_thread = E4M3_ALIGNMENT;
	const int threads = 128;
	const int vals_per_block = vals_per_thread * threads;
	const int blocks = static_cast<int>((numel + vals_per_block - 1) / vals_per_block);

	c10::Float8_e4m3fn *dst_f8 = dst.data_ptr<c10::Float8_e4m3fn>();
	__nv_fp8_e4m3 *dst_ptr = reinterpret_cast<__nv_fp8_e4m3*>(dst_f8);

	const float *scale_ptr = scale.data_ptr<float>();

	switch (src.scalar_type()) {
		case at::kHalf: {
			const at::Half *h = src.data_ptr<at::Half>();
			const __half *src_ptr = reinterpret_cast<const __half*>(h);
			quantize_e4m3_tensor_kernel<__half><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_ptr, numel);
			break;
		}
		case at::kBFloat16: {
			const at::BFloat16 *bh = src.data_ptr<at::BFloat16>();
			const __nv_bfloat16 *src_ptr = reinterpret_cast<const __nv_bfloat16*>(bh);
			quantize_e4m3_tensor_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_ptr, numel);
			break;
		}
		default:
			TORCH_CHECK(false, "Unsupported dtype for quantize_e4m3_tensor: ", src.scalar_type());
	}
}



void launch_quantize_nvfp4(const at::Tensor &src, at::Tensor &dst, at::Tensor &scale, at::Tensor &block_scale, cudaStream_t stream)
{
	const uint64_t numel = static_cast<uint64_t>(src.numel());
	const int vals_per_thread = NVFP4_BLOCK_SIZE;
	const int threads = 128;
	const int vals_per_block = vals_per_thread * threads;
	const int blocks = static_cast<int>((numel + vals_per_block - 1) / vals_per_block);

	void *dst_raw = dst.data_ptr();
	__nv_fp4x2_e2m1 *dst_ptr = reinterpret_cast<__nv_fp4x2_e2m1*>(dst_raw);

	const float *scale_ptr = scale.data_ptr<float>();

	c10::Float8_e4m3fn *block_scale_f8 = block_scale.data_ptr<c10::Float8_e4m3fn>();
	__nv_fp8_e4m3 *block_scale_ptr = reinterpret_cast<__nv_fp8_e4m3*>(block_scale_f8);

	switch (src.scalar_type()) {
		case at::kHalf: {
			const at::Half *h = src.data_ptr<at::Half>();
			const __half *src_ptr = reinterpret_cast<const __half*>(h);
			quantize_nvfp4_kernel<__half><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_ptr, block_scale_ptr, numel);
			break;
		}
		case at::kBFloat16: {
			const at::BFloat16 *bh = src.data_ptr<at::BFloat16>();
			const __nv_bfloat16 *src_ptr = reinterpret_cast<const __nv_bfloat16*>(bh);
			quantize_nvfp4_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_ptr, block_scale_ptr, numel);
			break;
		}
		default:
			TORCH_CHECK(false, "Unsupported dtype for quantize_nvfp4_kernel: ", src.scalar_type());
	}
}