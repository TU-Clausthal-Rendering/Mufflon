#include "material.hpp"
#include "core/memory/dyntype_memory.hpp"
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


char* IMaterial::get_descriptor(Device device, char* outBuffer) const {
	*as<MaterialDescriptorBase>(outBuffer) = MaterialDescriptorBase{
		m_type,
		get_properties(),
		m_innerMedium,
		m_outerMedium
	};
	return get_subdescriptor(device, outBuffer + sizeof(MaterialDescriptorBase));
}


}}} // namespace mufflon::scene::materials
