#include "core/cameras/camera.hpp"
#include "core/cameras/pinhole.hpp"
#include "core/cameras/focus.hpp"
#include "core/scene/descriptors.hpp"
#include <array>
#include <string>

namespace mufflon::cameras {

const std::string& to_string(CameraModel type) {
	static const std::array<std::string, int(CameraModel::NUM)> enumNames {
		"PINHOLE",
		"FOCUS",
		"ORTHOGRAPHIC"
	};

	return enumNames[int(type)];
}

static_assert(ei::max(sizeof(PinholeParams), sizeof(FocusParams)) == MAX_CAMERA_PARAM_SIZE,
	"MAX_CAMERA_PARAM_SIZE outdated please change the number in the header file.");

} // namespace mufflon::cameras