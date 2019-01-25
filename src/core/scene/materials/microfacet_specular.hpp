#pragma once

#include "material.hpp"
#include "microfacet_base.hpp"
#include "core/export/api.h"
#include "core/math/sampling.hpp"
#include "core/scene/textures/texture.hpp"
#include "core/scene/textures/interface.hpp"

namespace mufflon { namespace scene { namespace materials {

struct TorranceParameterPack {
	Spectrum albedo;
	float angle;
	ei::Vec2 roughness;
	NDF ndf;
};

template<Device dev>
struct TorranceDesc {
	textures::ConstTextureDevHandle_t<dev> albedoTex;
	textures::ConstTextureDevHandle_t<dev> roughnessTex;
	NDF ndf;

	CUDA_FUNCTION int fetch(const UvCoordinate& uvCoordinate, char* outBuffer) const {
		ei::Vec4 roughness = sample(roughnessTex, uvCoordinate);
		if(get_texture_channel_count(roughnessTex) == 1)
			roughness.y = roughness.x;
		*as<TorranceParameterPack>(outBuffer) = TorranceParameterPack{
			Spectrum{ sample(albedoTex, uvCoordinate) },
			roughness.z, {roughness.x, roughness.y},
			ndf
		};
		return sizeof(TorranceParameterPack);
	}
};

// Class for the handling of the Torrance-Sparrow microfacet reflection model.
class Torrance : public IMaterial {
public:
	Torrance(TextureHandle albedo, TextureHandle roughness, NDF ndf) :
		IMaterial{Materials::TORRANCE},
		m_albedo{albedo},
		m_roughness{roughness},
		m_ndf{ndf}
	{}

	MaterialPropertyFlags get_properties() const noexcept final {
		return MaterialPropertyFlags::REFLECTIVE | MaterialPropertyFlags::HALFVECTOR_BASED;
	}

	std::size_t get_descriptor_size(Device device) const final {
		std::size_t s = IMaterial::get_descriptor_size(device);
		device_switch(device, return sizeof(TorranceDesc<dev>) + s);
		return 0;
	}

	std::size_t get_parameter_pack_size() const final {
		return IMaterial::get_parameter_pack_size()
			+ sizeof(TorranceParameterPack);
	}

	char* get_subdescriptor(Device device, char* outBuffer) const final {
		device_switch(device,
			(*as<TorranceDesc<dev>>(outBuffer) =
				TorranceDesc<dev>{ m_albedo->acquire_const<dev>(),
								   m_roughness->acquire_const<dev>(),
								   m_ndf });
			return outBuffer + sizeof(TorranceDesc<dev>);
		);
		return nullptr;
	}

	Medium compute_medium() const final {
		// Use some average dielectric refraction index and a maximum absorption
		return Medium{ei::Vec2{1.3f, 0.0f}, Spectrum{std::numeric_limits<float>::infinity()}};
	}

	TextureHandle get_albedo() const noexcept {
		return m_albedo;
	}
	TextureHandle get_roughness() const noexcept {
		return m_albedo;
	}
	NDF get_ndf() const noexcept {
		return m_ndf;
	}
private:
	TextureHandle m_albedo;
	TextureHandle m_roughness;
	NDF m_ndf;
};

namespace mat_details {

}



// The importance sampling routine
CUDA_FUNCTION math::PathSample
torrance_sample(const TorranceParameterPack& params,
			   const Direction& incidentTS,
			   Boundary& boundary,
			   const math::RndSet2_1& rndSet) {
	// Importance sampling for the ndf
	math::DirectionSample cavityTS = sample_ndf(params.ndf, params.roughness, rndSet);

	// Find the visible half vector.
	float iDotH;
	Direction halfTS = sample_visible_normal_vcavity(incidentTS, cavityTS.direction, rndSet.i0, iDotH);
	// TODO rotate half vector

	boundary.set_halfTS(halfTS);

	// Reflect the vector 
	Direction excidentTS = (2.0f * iDotH) * halfTS - incidentTS;

	// Get geometry factors for PDF and throughput computation
	float ge = geoshadowing_vcavity(iDotH, excidentTS.z, halfTS.z, params.roughness);
	float gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
	float g = geoshadowing_vcavity_reflection(gi, ge);
	if(ge == 0.0f || gi == 0.0f) // Completely nullyfy the invalid result
		g = gi = ge = 0.0f;

	// Copy the sign for two sided diffuse
	return math::PathSample {
		Spectrum{params.albedo} * sdiv(g,gi),
		math::PathEventType::REFLECTED,
		excidentTS,
		cavityTS.pdf * sdiv(gi, ei::abs(4.0f * incidentTS.z * halfTS.z)),
		cavityTS.pdf * sdiv(ge, ei::abs(4.0f * excidentTS.z * halfTS.z))
	};
}

// The evaluation routine
CUDA_FUNCTION math::EvalValue
torrance_evaluate(const TorranceParameterPack& params,
				 const Direction& incidentTS,
				 const Direction& excidentTS,
				 Boundary& boundary) {
	using namespace mat_details;
	// No transmission
	if(incidentTS.z * excidentTS.z < 0.0f) return math::EvalValue{};

	// Geometry Term
	Direction halfTS = boundary.get_halfTS(incidentTS, excidentTS);
	float iDotH = dot(incidentTS, halfTS);
	float ge = geoshadowing_vcavity(iDotH, excidentTS.z, halfTS.z, params.roughness);
	float gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
	float g = geoshadowing_vcavity_reflection(gi, ge);

	// TODO: rotate halfTS

	// Normal Density Term
	float d = eval_ndf(params.ndf, params.roughness, halfTS);

	// Fresnel is done as layer blending...

	return math::EvalValue {
		params.albedo * sdiv(g * d, ei::abs(4.0f * incidentTS.z * excidentTS.z)),
		ei::abs(excidentTS.z),
		AngularPdf(sdiv(gi * d, ei::abs(4.0f * incidentTS.z))),
		AngularPdf(sdiv(ge * d, ei::abs(4.0f * excidentTS.z)))
	};
}

// The albedo routine
CUDA_FUNCTION Spectrum
torrance_albedo(const TorranceParameterPack& params) {
	return params.albedo;
}

}}} // namespace mufflon::scene::materials
