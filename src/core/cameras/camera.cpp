#include "core/cameras/camera.hpp"
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

} // namespace mufflon::cameras