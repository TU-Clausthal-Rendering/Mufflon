#include "allocator.hpp"
#include "synchronize.hpp"


namespace mufflon {
namespace memory_details {

__global__ void cuda_copy_element(const void* devElemMem, void* targetMem, const std::size_t elemBytes,
							 const std::size_t count) {
	for(std::size_t i = 0u; i < count; ++i)
		memcpy(targetMem, devElemMem, elemBytes);
}

void copy_element(const void* element, void* targetMem, const std::size_t elemBytes,
				  const std::size_t count) {
	void* deviceMem;
	cuda::check_error(cudaMalloc(&deviceMem, elemBytes));
	cudaMemcpy(deviceMem, targetMem, elemBytes, cudaMemcpyDefault);

	cuda_copy_element<<< 1, 1024 >>>(deviceMem, targetMem, elemBytes, count);

	cuda::check_error(cudaGetLastError());
	cuda::check_error(cudaFree(deviceMem));
}

} // namespace memory_details
} // namespace mufflon