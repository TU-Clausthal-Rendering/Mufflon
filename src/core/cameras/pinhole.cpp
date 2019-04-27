#include "pinhole.hpp"
#include "core/memory/residency.hpp"
#include "core/memory/synchronize.hpp"
#include "core/memory/dyntype_memory.hpp"

namespace mufflon::cameras {

	void Pinhole::get_parameter_pack(CameraParams* outBuffer, const Pixel& resolution) const {
		PinholeParams buffer{
			CameraParams{
				CameraModel::PINHOLE,
				std::numeric_limits<scene::materials::MediumHandle>::max() // Placeholder
			}, 
			m_position,
			m_tanVFov,
			get_view_dir(),
			m_near,
			get_up_dir(),
			m_far,
			ei::Vec<u16,2>{resolution}
		};
		copy(as<PinholeParams>(outBuffer), &buffer, 0, sizeof(PinholeParams));
	}

	std::size_t Pinhole::get_parameter_pack_size() const {
		return sizeof(PinholeParams);
	}

} // namespace mufflon::cameras
