#pragma once

#include "residency.hpp"

namespace mufflon { // There is no memory namespace on purpose

namespace synchronize_detail {

template < std::size_t I, Device dev, class Tuple, class T, class... Args >
void synchronize_impl(Tuple& tuple, util::DirtyFlags<Device>& flags,
					  T& sync, Args... args) {
	if constexpr(I < Tuple::size) {
		// Workaround for VS2017 bug: otherwise you may use the 'Type' template of the
		// tagged tuple
		auto& changed = tuple.template get<I>();
		constexpr Device CHANGED_DEVICE = std::decay_t<decltype(changed)>::DEVICE;
		if(flags.has_changes(CHANGED_DEVICE)) {
			synchronize(changed, sync, std::forward<Args>(args)...);
		} else {
			synchronize_impl<I + 1u, dev>(tuple, flags, sync);
		}
	}
}

} // namespace synchronize_detail

// Synchronizes changes from the tuple to the given class
template < Device dev, class Tuple, class T, class... Args >
void synchronize(Tuple& tuple, util::DirtyFlags<Device>& flags, T& sync, Args... args) {
	if(flags.needs_sync(dev)) {
		if(flags.has_competing_changes())
			throw std::runtime_error("Competing changes for attribute detected!");
		// Synchronize
		synchronize_detail::synchronize_impl<0u, dev>(tuple, flags, sync,
													  std::forward<Args>(args)...);
	}
}

// A number of copy primitives which call the internal required methods
// NOTE: There are synchronize() methods in the residency.hpp which have a similar
// functionallity. However, they are too specialized. MAYBE they can be replaced by
// this one
template < Device dev >
using DevPtr = void*;
template < Device dev >
using ConstDevPtr = const void*;
// TODO: OpenGL specialization
template < Device dstDev, Device srcDev >
inline void copy(DevPtr<dstDev> dst, ConstDevPtr<srcDev> src, std::size_t size ) {
	mAssertMsg(false, "Unimplemented copy specialization.");
}

template <>
inline void copy<Device::CPU, Device::CPU>(void* dst, const void* src, std::size_t size) {
	memcpy(dst, src, size);
}
template <>
inline void copy<Device::CUDA, Device::CPU>(void* dst, const void* src, std::size_t size) {
	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}
template <>
inline void copy<Device::CPU, Device::CUDA>(void* dst, const void* src, std::size_t size) {
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}
template <>
inline void copy<Device::CUDA, Device::CUDA>(void* dst, const void* src, std::size_t size) {
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}
// TODO: OpenGL (glBufferSubData with offset and object handle as target/src types

} // namespace mufflon