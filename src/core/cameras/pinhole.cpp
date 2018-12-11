#include "pinhole.hpp"
#include "core/memory/residency.hpp"
#include "core/memory/synchronize.hpp"
#include "core/memory/dyntype_memory.hpp"

namespace mufflon::cameras {

	void Pinhole::get_parameter_pack(CameraParams* outBuffer, const Pixel& resolution) const {
		PinholeParams buffer {
			CameraModel::PINHOLE,
			m_position,
			m_tanVFov,
			get_view_dir(),
			m_near,
			get_up_dir(),
			m_far,
			ei::Vec<u16,2>{resolution}
		};
		copy(as<PinholeParams>(outBuffer), &buffer, sizeof(PinholeParams));
	}

	std::size_t Pinhole::get_parameter_pack_size() const {
		return sizeof(PinholeParams);
	}

} // namespace mufflon::cameras
