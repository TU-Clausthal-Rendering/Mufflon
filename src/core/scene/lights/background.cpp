#include "background.hpp"
#include "texture_sampling.hpp"

namespace mufflon::scene::lights {

template< Device dev >
const BackgroundDesc<dev> Background::acquire_const() {
	switch(m_type) {
		case BackgroundType::COLORED:
			return BackgroundDesc<dev>{ {}, {}, m_type, m_color, m_flux };
		case BackgroundType::ENVMAP:
			mAssert(m_envLight != nullptr);
			if(!m_summedAreaTable) { // Flux and SAT are not computed?
				m_summedAreaTable = create_summed_area_table(m_envLight->acquire_const<Device::CPU>());
			// TODO: flux
			}
			return BackgroundDesc<dev>{
				m_envLight->acquire_const<dev>(),
				m_summedAreaTable->acquire_const<dev>(),
				m_type, m_color, m_flux
			};
		default: mAssert(false); return {};
	}
}

template const BackgroundDesc<Device::CPU> Background::acquire_const<Device::CPU>();
template const BackgroundDesc<Device::CUDA> Background::acquire_const<Device::CUDA>();

} // namespace mufflon::scene::lights
