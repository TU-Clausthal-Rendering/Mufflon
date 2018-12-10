#include "focus.hpp"
#include "core/memory/residency.hpp"
#include "core/memory/synchronize.hpp"

namespace mufflon::cameras {

void Focus::get_parameter_pack(CameraParams* outBuffer, Device dev, const Pixel& resolution) const {
	FocusParams buffer{
		CameraModel::PINHOLE,
		m_position,
		m_tanVFov,
		get_view_dir(),
		m_near,
		get_up_dir(),
		m_far,
		m_sensorHeight,
		m_focalDistance,
		m_lensRadius,
		ei::Vec<u16,2>{resolution}
	};
	switch(dev) {
		case Device::CPU: copy<Device::CPU, Device::CPU>(outBuffer, &buffer, sizeof(FocusParams)); break;
		case Device::CUDA: copy<Device::CUDA, Device::CPU>(outBuffer, &buffer, sizeof(FocusParams)); break;
		//case Device::OPENGL: copy<, Device::CPU>(outBuffer, &buffer, sizeof(PinholeParams)); break;
	}
}

std::size_t Focus::get_parameter_pack_size() const {
	return sizeof(FocusParams);
}

} // namespace mufflon::cameras
