#include "pinhole.hpp"
#include "core/memory/synchronize.hpp"

namespace mufflon::cameras {

	void Pinhole::get_parameter_pack(CameraParams* outBuffer, Device dev) const {
		PinholeParams buffer {
			CameraModel::PINHOLE,
			m_position,
			get_x_dir(),
			m_near,
			get_up_dir(),
			m_far,
			get_view_dir(),
			m_tanVFov
		};
		switch(dev) {
			case Device::CPU: copy<Device::CPU, Device::CPU>(outBuffer, &buffer, sizeof(PinholeParams)); break;
			case Device::CUDA: copy<Device::CUDA, Device::CPU>(outBuffer, &buffer, sizeof(PinholeParams)); break;
			//case Device::OPENGL: copy<, Device::CPU>(outBuffer, &buffer, sizeof(PinholeParams)); break;
		}
	}

	std::size_t Pinhole::get_parameter_pack_size() const {
		return sizeof(PinholeParams);
	}

} // namespace mufflon::cameras
