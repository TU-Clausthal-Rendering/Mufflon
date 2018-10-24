#include "material.hpp"

namespace mufflon { namespace scene { namespace material {

const std::string& to_string(Materials type)
{
	static const std::array<std::string, int(Materials::NUM)> enumNames {
		"LAMBERT",
		"TORRANCE",
		"WALTER",
		"EMISSIVE",
		"ORENNAYAR",
		"BLEND",
		"FRESNEL",
		"GLASS"
	};

	return enumNames[int(type)];
}

}}} // namespace mufflon::scene::material