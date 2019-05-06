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
	return sizeof(MaterialDescriptorBase) + sizeof(SubMaterial::SampleType);
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
			tex[i] = m_textures[i]->acquire_const<dev>();
		char* subParams = as<char>(desc + 1) + sizeof(textures::ConstTextureDevHandle_t<dev>) * int(SubMaterial::TEX_COUNT);
		*as<SubMaterial::NonTexParams>(subParams) = m_material.nonTexParams;
		return subParams + (std::is_empty<SubMaterial::NonTexParams>::value ? 0 : sizeof(SubMaterial::NonTexParams));
	);
	return as<char>(desc + 1);
}

template<Materials M>
Emission Material<M>::get_emission() const {
	if constexpr(details::has_emission<SubMaterial>(0)) {
		return m_material.get_emission(m_textures, 0);
	} else {
		return IMaterial::get_emission();
	}
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

// C++14 incompatible member functions of blend models
template<class LayerA, class LayerB>
Emission MatBlend<LayerA, LayerB>::get_emission(const TextureHandle* texTable, int texOffset) const {
	if constexpr(details::has_emission<LayerA>(0)) {
		Emission em = layerA.get_emission(texTable, texOffset);
		em.scale *= nonTexParams.factorA;
		return em;
	} else if constexpr(details::has_emission<LayerB>(0)) {
		Emission em = layerB.get_emission(texTable, texOffset+LayerA::TEX_COUNT);
		em.scale *= nonTexParams.factorB;
		return em;
	} else return Emission{nullptr, Spectrum{0.0f}};
}

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
Emission MatBlendFresnel<LayerA, LayerB>::get_emission(const TextureHandle* texTable, int texOffset) const {
	if constexpr(details::has_emission<LayerA>(0)) {
		Emission em = layerA.get_emission(texTable, texOffset);
		// TODO: attenuate with F, requires the directions!
		return em;
	} else if constexpr(details::has_emission<LayerB>(0)) {
		// TODO: attenuate with 1-F, requires the directions!
		Emission em = layerB.get_emission(texTable, texOffset+LayerA::TEX_COUNT);
		return em;
	} else return Emission{nullptr, Spectrum{0.0f}};
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
	p->get_emission();
	p->compute_medium();
	//(void)std::declval<Material<PREV_MAT>>();
	instanciate_materials<PREV_MAT>();
}
template<>
constexpr void instanciate_materials<Materials(0)>() {}
template void instanciate_materials<Materials::NUM>();

//template Material<Materials::LAMBERT>;

} // namespace mufflon::scene::materials
