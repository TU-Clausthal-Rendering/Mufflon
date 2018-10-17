#include <cuda_runtime.h>
#include <iostream>

namespace cuda {

	void test() {
		std::cout << "Hello CUDA!" << std::endl;
		int device_count;
		cudaGetDeviceCount(&device_count);
		std::cout << "Devices: " << device_count << std::endl;
	}

} // namespace cuda