#pragma once

#include "material.hpp"
#include "microfacet_base.hpp"
#include "core/export/api.h"
#include "core/memory/dyntype_memory.hpp"
#include "core/math/sampling.hpp"
#include "core/scene/textures/texture.hpp"
#include "core/scene/textures/interface.hpp"

namespace mufflon { namespace scene { namespace materials {

struct WalterParameterPack {
	Spectrum absorption; // Absorption λ per meter (transmission = exp(-λ*d))
	float angle;
	ei::Vec2 roughness;
	NDF ndf;
};

template<Device dev>
struct WalterDesc {
	textures::ConstTextureDevHandle_t<dev> roughnessTex;
	Spectrum absorption;
	NDF ndf;

	CUDA_FUNCTION int fetch(const UvCoordinate& uvCoordinate, char* outBuffer) const {
		ei::Vec4 roughness = sample(roughnessTex, uvCoordinate);
		if(get_texture_channel_count(roughnessTex) == 1)
			roughness.y = roughness.x;
		*as<WalterParameterPack>(outBuffer) = WalterParameterPack{
			absorption,
			roughness.z, {roughness.x, roughness.y},
			ndf
		};
		return sizeof(WalterParameterPack);
	}
};

// Class for the handling of the Walter microfacet refraction model.
class Walter : public IMaterial {
public:
	Walter(Spectrum absorption, TextureHandle roughness, float refractionIndex, NDF ndf) :
		IMaterial{Materials::WALTER},
		m_absorption{absorption},
		m_roughness{roughness},
		m_refractionIndex{refractionIndex},
		m_ndf{ndf}
	{}

	MaterialPropertyFlags get_properties() const noexcept final {
		return MaterialPropertyFlags::REFRACTIVE | MaterialPropertyFlags::HALFVECTOR_BASED;
	}

	std::size_t get_descriptor_size(Device device) const final {
		std::size_t s = IMaterial::get_descriptor_size(device);
		device_switch(device, return sizeof(WalterDesc<dev>) + s);
		return 0;
	}

	std::size_t get_parameter_pack_size() const final {
		return IMaterial::get_parameter_pack_size()
			+ sizeof(WalterParameterPack);
	}

	char* get_subdescriptor(Device device, char* outBuffer) const final {
		device_switch(device,
			(*as<WalterDesc<dev>>(outBuffer) =
				WalterDesc<dev>{ m_roughness->acquire_const<dev>(),
								 m_absorption,
								 m_ndf });
			return outBuffer + sizeof(WalterDesc<dev>);
		);
		return nullptr;
	}

	Medium compute_medium() const final {
		return Medium{ei::Vec2{m_refractionIndex, 0.0f}, m_absorption};
	}

	Spectrum get_absorption() const noexcept {
		return m_absorption;
	}
	TextureHandle get_roughness() const noexcept {
		return m_roughness;
	}
	float get_refraction_index() const noexcept {
		return m_refractionIndex;
	}
	NDF get_ndf() const noexcept {
		return m_ndf;
	}
private:
	TextureHandle m_roughness;
	Spectrum m_absorption;
	float m_refractionIndex;
	NDF m_ndf;
};

// The importance sampling routine
CUDA_FUNCTION math::PathSample
walter_sample(const WalterParameterPack& params,
			  const Direction& incidentTS,
			  Boundary& boundary,
			  const math::RndSet2_1& rndSet,
			  bool adjoint) {
	// Importance sampling for the ndf
	math::DirectionSample cavityTS = sample_ndf(params.ndf, params.roughness, rndSet);

	// Find the visible half vector.
	float iDotH;
	Direction halfTS = sample_visible_normal_vcavity(incidentTS, cavityTS.direction, rndSet.i0, iDotH);
	// TODO rotate half vector

	boundary.set_halfTS(halfTS);

	// Compute the refraction index
	float n_i = boundary.incidentMedium.get_refraction_index().x;
	float n_e = boundary.otherMedium.get_refraction_index().x;
	float eta = n_i / n_e;
	float etaSq = eta * eta;
	float iDotHabs = ei::abs(iDotH);
	// Snells law
	float eDotHabs = sqrt(ei::max(0.0f, 1.0f - etaSq * (1.0f - iDotHabs * iDotHabs)));
	float eDotH = eDotHabs * -ei::sgn(iDotH); // Opposite to iDotH
	// The refraction vector
	Direction excidentTS = ei::sgn(iDotH) * (eta * iDotHabs - eDotHabs) * halfTS - eta * incidentTS;

	// Get geometry and common factors for PDF and throughput computation
	float ge = geoshadowing_vcavity(eDotH, excidentTS.z, halfTS.z, params.roughness);
	float gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
	float g = geoshadowing_vcavity_transmission(gi, ge);
	if(ge == 0.0f || gi == 0.0f) // Completely nullyfy the invalid result
		g = gi = ge = 0.0f;

	AngularPdf common = cavityTS.pdf * sdiv(iDotHabs * eDotHabs, ei::sq(n_i * iDotH + n_e * eDotH) * halfTS.z);
	Spectrum throughput { sdiv(g, gi) };
	if(!adjoint)
		throughput *= etaSq;

	return math::PathSample {
		throughput,
		math::PathEventType::REFRACTED,
		excidentTS,
		common * sdiv(gi * n_e * n_e, ei::abs(incidentTS.z)),
		common * sdiv(ge * n_i * n_i, ei::abs(excidentTS.z)),
	};
}

// The evaluation routine
CUDA_FUNCTION math::EvalValue
walter_evaluate(const WalterParameterPack& params,
				const Direction& incidentTS,
				const Direction& excidentTS,
				Boundary& boundary,
				bool adjoint) {
	// No reflection
	if(incidentTS.z * excidentTS.z > 0.0f) return math::EvalValue{};

	// General terms. For refraction iDotH != eDotH!
	Direction halfTS = boundary.get_halfTS(incidentTS, excidentTS);
	float iDotH = dot(incidentTS, halfTS);
	float eDotH = dot(excidentTS, halfTS);
	float n_i = boundary.incidentMedium.get_refraction_index().x;
	float n_e = boundary.otherMedium.get_refraction_index().x;

	// Geometry Term
	float ge = geoshadowing_vcavity(eDotH, excidentTS.z, halfTS.z, params.roughness);
	float gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
	float g = geoshadowing_vcavity_transmission(gi, ge);

	// Normal Density Term
	float d = eval_ndf(params.ndf, params.roughness, halfTS);

	// Fresnel is done as layer blending...

	float common = sdiv(ei::abs(d * iDotH * eDotH), ei::sq(n_i * iDotH + n_e * eDotH));
	// To make the BSDF reciprocal, the n_i/n_e must be exchanged for the adjoint.
	float n_x = adjoint ? n_e : n_i;
	float bsdf = g * common * sdiv(n_x * n_x, ei::abs(incidentTS.z * excidentTS.z));
	return math::EvalValue {
		Spectrum{bsdf},
		ei::abs(excidentTS.z),
		AngularPdf(gi * common * sdiv(n_e * n_e, ei::abs(incidentTS.z))),
		AngularPdf(ge * common * sdiv(n_i * n_i, ei::abs(excidentTS.z)))
	};
}

// The albedo routine
CUDA_FUNCTION Spectrum
walter_albedo(const WalterParameterPack& params) {
	// Compute a pseudo value based on the absorption.
	// The problem: the true amount of transmittance depends on the depth of the medium.
	return 1.0f / (Spectrum{1.0f} + params.absorption);
}

}}} // namespace mufflon::scene::materials
