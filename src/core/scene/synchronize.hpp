#pragma once

#include "residency.hpp"

namespace mufflon { namespace scene {

namespace synchronize_detail {

template < std::size_t I, Device dev, class Tuple, class T, class... Args >
void synchronize_impl(Tuple& tuple, util::DirtyFlags<Device>& flags,
					  T& sync, Args... args) {
	if constexpr(I < Tuple::size) {
		// Workaround for VS2017 bug: otherwise you may use the 'Type' template of the
		// tagged tuple
		auto& changed = tuple.get<I>();
		constexpr Device CHANGED_DEVICE = std::decay_t<decltype(changed)>::DEVICE;
		if(flags.has_changes(CHANGED_DEVICE)) {
			synchronize(changed, sync, std::forward<Args>(args)...);
		} else {
			synchronize_impl<I + 1u, dev>(tuple, flags, sync);
		}
	}
}

} // namespace synchronize_detail

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

}} // namespace mufflon::scene