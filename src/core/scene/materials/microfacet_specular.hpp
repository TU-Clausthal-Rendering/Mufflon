#pragma once

#include "core/export/api.h"
#include "core/math/sampling.hpp"
#include "core/scene/textures/interface.hpp"
#include "material_definitions.hpp"
#include "microfacet_base.hpp"

namespace mufflon { namespace scene { namespace materials {

inline CUDA_FUNCTION MatSampleTorrance fetch(const textures::ConstTextureDevHandle_t<CURRENT_DEV>* textures,
									  const ei::Vec4* texValues,
									  int texOffset,
									  const typename MatTorrance::NonTexParams& params) {
	ei::Vec2 roughness { texValues[MatTorrance::ROUGHNESS+texOffset].x };
	if(get_texture_channel_count(textures[MatTorrance::ROUGHNESS+texOffset]) > 1)
		roughness.y = texValues[MatTorrance::ROUGHNESS+texOffset].y;
	return MatSampleTorrance{
		Spectrum{texValues[MatTorrance::ALBEDO+texOffset]},
		params.shadowing, params.ndf, roughness
	};
}


// The importance sampling routine
inline CUDA_FUNCTION math::PathSample sample(const MatSampleTorrance& params,
									  const Direction& incidentTS,
									  Boundary& boundary,
									  const math::RndSet2_1& rndSet,
									  bool /*adjoint*/) {
	float iDotH;
	Direction halfTS;
	AngularPdf cavityPdf;
	u64 rnd = rndSet.i0;
	if(params.shadowing == ShadowingModel::SMITH) {
		halfTS = sample_visible_normal_smith(params.ndf, incidentTS, params.roughness, rndSet, rnd);
		cavityPdf = AngularPdf(eval_ndf(params.ndf, params.roughness, halfTS));
		cavityPdf *= halfTS.z;
		iDotH = ei::dot(incidentTS, halfTS);
	} else {
		// Importance sampling for the ndf
		math::DirectionSample cavityTS = sample_ndf(params.ndf, params.roughness, rndSet);

		// Find the visible half vector.
		auto h = sample_visible_normal_vcavity(incidentTS, cavityTS.direction, rnd);
		halfTS = h.halfTS;
		iDotH = h.cosI;
		cavityPdf = cavityTS.pdf;
	}
	boundary.set_halfTS(halfTS);
	mAssert(halfTS.z > 0);

	// Reflect the vector 
	Direction excidentTS = (2.0f * iDotH) * halfTS - incidentTS;

	// Get geometry factors for PDF and throughput computation
	float ge, gi, g;
	if(params.shadowing == ShadowingModel::SMITH) {
		ge = geoshadowing_smith(iDotH, excidentTS, params.roughness, params.ndf);
		gi = geoshadowing_smith(iDotH, incidentTS, params.roughness, params.ndf);
		g = geoshadowing_smith_reflection(gi, ge);
	} else {
		ge = geoshadowing_vcavity(iDotH, excidentTS.z, halfTS.z, params.roughness);
		gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
		g = geoshadowing_vcavity_reflection(gi, ge);
	}
	if(ge == 0.0f || gi == 0.0f) // Completely nullify the invalid result
		return math::PathSample {};

	// Copy the sign for two sided diffuse
	return math::PathSample {
		params.albedo * sdiv(g, gi),
		math::PathEventType::REFLECTED,
		excidentTS,
		{ cavityPdf * sdiv(gi, ei::abs(4.0f * incidentTS.z * halfTS.z)),
		  cavityPdf * sdiv(ge, ei::abs(4.0f * excidentTS.z * halfTS.z)) }
	};
}

// The evaluation routine
inline CUDA_FUNCTION math::BidirSampleValue evaluate(const MatSampleTorrance& params,
											  const Direction& incidentTS,
											  const Direction& excidentTS,
											  Boundary& boundary) {
	// No transmission
	if(incidentTS.z * excidentTS.z < 0.0f) return math::BidirSampleValue{};

	// Geometry Term
	Direction halfTS = boundary.get_halfTS(incidentTS, excidentTS);
	float iDotH = dot(incidentTS, halfTS);
	float ge, gi, g;
	if(params.shadowing == ShadowingModel::SMITH) {
		ge = geoshadowing_smith(iDotH, excidentTS, params.roughness, params.ndf);
		gi = geoshadowing_smith(iDotH, incidentTS, params.roughness, params.ndf);
		g = geoshadowing_smith_reflection(gi, ge);
	} else {
		 ge = geoshadowing_vcavity(iDotH, excidentTS.z, halfTS.z, params.roughness);
		 gi = geoshadowing_vcavity(iDotH, incidentTS.z, halfTS.z, params.roughness);
		 g = geoshadowing_vcavity_reflection(gi, ge);
	}

	if(ge == 0.0f || gi == 0.0f)
		return math::BidirSampleValue {};

	// Normal Density Term
	float d = eval_ndf(params.ndf, params.roughness, halfTS);

	// Fresnel is done as layer blending...

	return math::BidirSampleValue {
		params.albedo * sdiv(g * d, ei::abs(4.0f * incidentTS.z * excidentTS.z)),
		{ AngularPdf(sdiv(gi * d, ei::abs(4.0f * incidentTS.z))),
		  AngularPdf(sdiv(ge * d, ei::abs(4.0f * excidentTS.z))) }
	};
}

// The albedo routine
inline CUDA_FUNCTION Spectrum albedo(const MatSampleTorrance& params) {
	return params.albedo;
}

inline CUDA_FUNCTION math::SampleValue emission(const MatSampleTorrance& /*params*/, const scene::Direction& /*geoN*/, const scene::Direction& /*excident*/) {
	return math::SampleValue{};
}

inline CUDA_FUNCTION float pdf_max(const MatSampleTorrance& params) {
	switch(params.ndf) {
		case NDF::BECKMANN:
			if(params.roughness < 1.f / std::sqrt(2.f))
				return 1.f / (ei::PI * params.roughness.x * params.roughness.y);
			else
				return (4.f * params.roughness.x * params.roughness.y + 2.f*(ei::sq(ei::E) - 1.f)) / (ei::PI * ei::sq(ei::E));
		case NDF::GGX:
			return 1.f / (ei::PI * params.roughness.x * params.roughness.y);
		default:
			return 0.f;
	}
}

template class MaterialSampleConcept<MatSampleTorrance>;
template class MaterialConcept<MatTorrance>;

}}} // namespace mufflon::scene::materials
