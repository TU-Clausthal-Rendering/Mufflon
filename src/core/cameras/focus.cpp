#include "focus.hpp"
#include "core/memory/residency.hpp"
#include "core/memory/synchronize.hpp"
#include "core/memory/dyntype_memory.hpp"

namespace mufflon::cameras {

void Focus::get_parameter_pack(CameraParams* outBuffer, const Pixel& resolution) const {
	// We treat the near plane as a scaling factor for everything: focal length, focus distance etc.
	FocusParams buffer{
		CameraParams{
			CameraModel::FOCUS,
			std::numeric_limits<scene::materials::MediumHandle>::max()// Placeholder
		},
		m_position,
		m_tanVFov,
		get_view_dir(),
		m_far,
		get_up_dir(),
		__float2half(m_sensorHeight * m_near / 2.f),
		__float2half(m_focusDistance * m_focalLength / (m_focusDistance + m_focalLength)),
		m_focusDistance * m_near,
		m_focalLength * m_near,
		m_lensRadius * m_near,
		ei::Vec<u16,2>{resolution}
	};
	copy(as<FocusParams>(outBuffer), &buffer, 0, sizeof(FocusParams));
}

std::size_t Focus::get_parameter_pack_size() const {
	return sizeof(FocusParams);
}

} // namespace mufflon::cameras
