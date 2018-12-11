#include "attribute.hpp"

namespace mufflon::scene {

// Synchronization specialization from CPU to CUDA (owning)
template <>
void AttributePool<Device::CPU>::synchronize<Device::CUDA>(AttributePool<Device::CUDA>& pool) {
	pool.make_present();
	cudaMemcpy(pool.get_pool_data(), this->get_pool_data(), this->get_byte_count(), cudaMemcpyDefault);
}

// Synchronization specialization from CUDA to CPU (owning)
template <>
void AttributePool<Device::CUDA>::synchronize<Device::CPU>(AttributePool<Device::CPU>& pool) {
	pool.make_present();
	cudaMemcpy(pool.get_pool_data(), this->get_pool_data(), this->get_byte_count(), cudaMemcpyDefault);
}

} // namespace mufflon::scene