#include "allocator.hpp"
#include "synchronize.hpp"
#include <device_launch_parameters.h>


namespace mufflon {
namespace memory_details {

__global__ void cuda_copy_element(const void* srcMem, void* dstMem, const std::size_t elemBytes,
							 const std::size_t count) {
	if(threadIdx.x == 0 && threadIdx.y == 0)
	for(std::size_t i = 0u; i < count; ++i)
		memcpy(dstMem, srcMem, elemBytes);
}

// Element is the (host-side) source, targetMem the (device-side) destination
void copy_element(const void* element, void* targetMem, const std::size_t elemBytes,
				  const std::size_t count) {
	void* deviceMem;
	cuda::check_error(cudaMalloc(&deviceMem, elemBytes));
	cudaMemcpy(deviceMem, element, elemBytes, cudaMemcpyDefault);

	cuda_copy_element<<< 1, 1024 >>>(deviceMem, targetMem, elemBytes, count);

	cuda::check_error(cudaGetLastError());
	cuda::check_error(cudaFree(deviceMem));
}

} // namespace memory_details
} // namespace mufflon