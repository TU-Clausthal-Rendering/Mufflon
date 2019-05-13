#include "material.hpp"
#include "core/memory/dyntype_memory.hpp"
#include "core/scene/textures/texture.hpp"
#include <array>
#include <string>

namespace mufflon::scene::materials {

const std::string& to_string(Materials type)
{
	static const std::array<std::string, int(Materials::NUM)> enumNames {
		"LAMBERT",
		"TORRANCE",
		"WALTER",
		"EMISSIVE",
		"ORENNAYAR",
		"LAMBERT_EMISSIVE"
	};

	return enumNames[int(type)];
}


template<Materials M>
MaterialPropertyFlags Material<M>::get_properties() const noexcept {
	return SubMaterial::PROPERTIES;
}

template<Materials M>
std::size_t Material<M>::get_parameter_pack_size() const {
	return sizeof(MaterialDescriptorBase) + sizeof(typename SubMaterial::SampleType);
}

template<Materials M>
char* Material<M>::get_descriptor(Device device, char* outBuffer) const {
	mAssertMsg(intptr_t(outBuffer) % 8 == 0, "Descriptors must be 8-byte aligned!");
	auto desc = as<MaterialDescriptorBase>(outBuffer);
	desc->type = get_type();
	desc->flags = SubMaterial::PROPERTIES;
	desc->innerMedium = m_innerMedium;
	desc->outerMedium = m_outerMedium;
	device_switch(device,
		textures::ConstTextureDevHandle_t<dev>* tex = as<textures::ConstTextureDevHandle_t<dev>>(desc + 1);
		for(int i = 0; i < int(SubMaterial::TEX_COUNT); ++i)
			tex[i] = m_textures[i]->template acquire_const<dev>();
		char* subParams = as<char>(desc + 1) + sizeof(textures::ConstTextureDevHandle_t<dev>) * int(SubMaterial::TEX_COUNT);
		*as<typename SubMaterial::NonTexParams>(subParams) = m_material.nonTexParams;
		return subParams + (std::is_empty<typename SubMaterial::NonTexParams>::value ? 0 : sizeof(typename SubMaterial::NonTexParams));
	);
	return as<char>(desc + 1);
}

template<Materials M>
Medium Material<M>::compute_medium() const {
	if constexpr(details::has_dependent_medium<SubMaterial>(0)) {
		return m_material.compute_medium();
	} else {
		// Use some average dielectric refraction index and a maximum absorption
		return Medium{ei::Vec2{1.3f, 0.0f}, Spectrum{std::numeric_limits<float>::infinity()}};
	}
}

// C++14 incompatible member functions of blend models (if constexpr -> not in header)
template<class LayerA, class LayerB>
Medium MatBlend<LayerA, LayerB>::compute_medium() const {
	if constexpr(details::has_dependent_medium<LayerA>(0))
		return layerA.compute_medium();
	else if constexpr(details::has_dependent_medium<LayerB>(0))
		return layerB.compute_medium();
	// Use some average dielectric refraction index and a maximum absorption
	return Medium{ei::Vec2{1.3f, 0.0f}, Spectrum{std::numeric_limits<float>::infinity()}};
}

template<class LayerA, class LayerB>
Medium MatBlendFresnel<LayerA, LayerB>::compute_medium() const {
	Medium baseMedium;
	if constexpr(details::has_dependent_medium<LayerA>(0))
		baseMedium = layerA.compute_medium();
	else if constexpr(details::has_dependent_medium<LayerB>(0))
		baseMedium = layerB.compute_medium();
	return Medium{ior, baseMedium.get_absorption_coeff()};
}




// Automatic instanciation of all defined materials
// Notice: this only works for MSVC (for e.g. clang-cl, they need to be explicitly instantiated;
// if necessary, update the list below)
template<Materials M>
constexpr void instanciate_materials() {
	constexpr Materials PREV_MAT = Materials(int(M)-1);
	// Instance without knowing the constructor
	//auto instance = *((Material<PREV_MAT>*)(nullptr));
	//(void)instance;
//	*((Material<PREV_MAT>*)(nullptr));
	Material<PREV_MAT>* p = (Material<PREV_MAT>*)1;
	p->get_properties();
	p->get_parameter_pack_size();
	p->get_descriptor(Device::CPU, nullptr);
	p->compute_medium();
	//(void)std::declval<Material<PREV_MAT>>();
	instanciate_materials<PREV_MAT>();
}
template<>
constexpr void instanciate_materials<Materials(0)>() {}
template void instanciate_materials<Materials::NUM>();

/*template class Material<Materials::EMISSIVE>;
template class Material<Materials::LAMBERT>;
template class Material<Materials::ORENNAYAR>;
template class Material<Materials::TORRANCE>;
template class Material<Materials::WALTER>;
template class Material<Materials::LAMBERT_EMISSIVE>;
template class Material<Materials::TORRANCE_LAMBERT>;
template class Material<Materials::FRESNEL_TORRANCE_LAMBERT>;
template class Material<Materials::WALTER_TORRANCE>;
template class Material<Materials::FRESNEL_TORRANCE_WALTER>;
template class Material<Materials::MICROFACET>;*/

} // namespace mufflon::scene::materials
