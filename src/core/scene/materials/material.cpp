#include "material.hpp"
#include <array>
#include <string>

namespace mufflon { namespace scene { namespace materials {

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


MaterialPropertyFlags IMaterial::get_property_flags() const noexcept {
	MaterialPropertyFlags flags;
	if(is_emissive()) flags.set(MaterialPropertyFlags::EMISSIVE);
	if(is_brdf()) flags.set(MaterialPropertyFlags::REFLECTIVE);
	if(is_btdf()) flags.set(MaterialPropertyFlags::REFRACTIVE);
	if(is_halfvector_based()) flags.set(MaterialPropertyFlags::HALFVECTOR_BASED);
	return flags;
}


}}} // namespace mufflon::scene::materials