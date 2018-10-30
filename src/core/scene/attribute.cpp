#include "attribute.hpp"

namespace mufflon::scene {

// Synchronization specialization from CPU to CUDA (owning)
template <>
void AttributePool<Device::CPU, true>::synchronize<Device::CUDA, true>(AttributePool<Device::CUDA, true>& pool) {
	pool.make_present();
	cudaMemcpy(pool.get_pool_data(), this->get_pool_data(), this->get_byte_count(), cudaMemcpyHostToDevice);
}

// Synchronization specialization from CPU to CUDA (non-owning)
template <>
void AttributePool<Device::CPU, false>::synchronize<Device::CUDA, true>(AttributePool<Device::CUDA, true>& pool) {
	std::size_t currOffset = 0u;
	pool.make_present();

	// Loop to copy the attributes
	for(std::size_t i = 0u; i < m_attributes.size(); ++i) {
		auto attrib = m_attributes[i];
		if(attrib != nullptr) {
			// Copy the current attribute into the buffer
			const std::size_t currLength = attrib->size_of();
			// TODO: this is hella UB, but I can't think of a better way so far...
			auto& castedProp = *reinterpret_cast<OpenMesh::PropertyT<char>*>(attrib);
			cudaMemcpy(&pool.get_pool_data()[currOffset], castedProp.data_vector().data(), currLength, cudaMemcpyHostToDevice);
			currOffset += currLength;
		}
	}
}

// Synchronization specialization from CUDA to CPU (owning)
template <>
void AttributePool<Device::CUDA, true>::synchronize<Device::CPU, true>(AttributePool<Device::CPU, true>& pool) {
	pool.make_present();
	cudaMemcpy(pool.get_pool_data(), this->get_pool_data(), this->get_byte_count(), cudaMemcpyDeviceToHost);
}

// Synchronization specialization from CUDA to CPU (non-owning)
template <>
void AttributePool<Device::CUDA, true>::synchronize<Device::CPU, false>(AttributePool<Device::CPU, false>& pool) {
	// TODO: make present
	std::size_t currOffset = 0u;

	// Loop to copy the attributes
	for(std::size_t i = 0u; i < pool.m_attributes.size(); ++i) {
		auto attrib = pool.m_attributes[i];
		if(attrib != nullptr) {
			// Copy from the contiguous buffer into the attributes
			const std::size_t currLength = attrib->size_of();
			// TODO: this is hella UB, but I can't think of a better way so far...
			auto& castedProp = *reinterpret_cast<OpenMesh::PropertyT<char>*>(attrib);
			cudaMemcpy(castedProp.data_vector().data(), &this->get_pool_data()[currOffset], currLength, cudaMemcpyDeviceToHost);
			currOffset += currLength;
		}
	}

	throw std::runtime_error("Synchronize between CPU and CUDA not yet implemented");
}

} // namespace mufflon::scene