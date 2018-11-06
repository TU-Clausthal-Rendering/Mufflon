#include "pinhole.hpp"

namespace mufflon::cameras {

	void Pinhole::get_parameter_pack(CameraParams& outBuffer) const {
		as<PinholeParams>(outBuffer) = PinholeParams{
			CameraModel::PINHOLE,
			m_position,
			get_x_dir(),
			m_near,
			get_up_dir(),
			m_far,
			get_view_dir(),
			m_tanVFov
		};
	}

	std::size_t Pinhole::get_parameter_pack_size() const {
		return sizeof(PinholeParams);
	}

} // namespace mufflon::cameras
