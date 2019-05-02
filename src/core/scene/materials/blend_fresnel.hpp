#pragma once

#include "core/export/api.h"
#include "core/math/sampling.hpp"
#include "material_definitions.hpp"

namespace mufflon { namespace scene { namespace materials {

struct Refraction {
	float f;
	float cosTAbs;
};

// Dielectric-Dielectric Fresnel for unpolarized light
// etaSq: (n_i / n_t)^2
CUDA_FUNCTION Refraction fresnel_dielectric(float n_i, float n_t, float cosIAbs) {
	float eta = ei::sq(n_i / n_t);
	float cosTAbs = sqrt(ei::max(0.0f, 1.0f - eta * (1.0f - cosIAbs * cosIAbs)));
	float rParl = sdiv(n_t * cosIAbs - n_i * cosTAbs, n_t * cosIAbs + n_i * cosTAbs);
	float rPerp = sdiv(n_t * cosTAbs - n_i * cosIAbs, n_t * cosTAbs + n_i * cosIAbs);
	return {0.5f * (rParl * rParl + rPerp * rPerp), cosTAbs};
}

// Dielectric-Conductor Fresnel for unpolarized light
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/#more-1921
// TODO: compute per frequency? Requires RGB n_i and n_t
CUDA_FUNCTION Refraction fresnel_conductor(ei::Vec2 n_i, ei::Vec2 n_t, float cosIAbs) {
	mAssertMsg(n_i.y == 0.0f, "Incident medium must be dielectric.");
	ei::Vec2 etaSq = sq(n_t / n_i.x);
	float cosISq = cosIAbs * cosIAbs;
	float sinISq = 1.0f - cosISq;
	float t0 = etaSq.x - etaSq.y - sinISq;
	float asq_bsq = sqrt(4.0f * etaSq.x * etaSq.y + t0 * t0);
	float a = sqrt(0.5f * (asq_bsq + t0));
	float t1 = 2.0f * a * cosIAbs;
	float rPerp = (asq_bsq + cosISq - t1) / (asq_bsq + cosISq + t1);
	t1 *= sinISq;
	float t2 = cosISq * asq_bsq + sinISq * sinISq;
	float rParl = rPerp * (t1 - t2) / (t1 + t2);
	return {0.5f * (rParl + rPerp), 0.0f};
}

template<class LayerA, class LayerB>
CUDA_FUNCTION typename MatBlendFresnel<LayerA, LayerB>::SampleType
fetch(const textures::ConstTextureDevHandle_t<CURRENT_DEV>* textures,
	  const ei::Vec4* texValues,
	  int texOffset,
	  const MatNTPBlendFresnel<LayerA, LayerB>& params) {
	return typename MatBlendFresnel<LayerA, LayerB>::SampleType{
		fetch(textures, texValues, texOffset, params.a),
		fetch(textures, texValues, texOffset+LayerA::TEX_COUNT, params.b)
	};
}

// The importance sampling routine
template<class LayerASample, class LayerBSample>
CUDA_FUNCTION math::PathSample sample(const MatSampleBlendFresnel<LayerASample, LayerBSample>& params,
									  const Direction& incidentTS,
									  Boundary& boundary,
									  math::RndSet2_1 rndSet,
									  bool adjoint) {
	// Determine a probability for each layer.
	// Note that f is determined with respect to the macroscopic normal.
	// A microfacet is not known yet.
	float n_i = boundary.incidentMedium.get_refraction_index().x;
	float n_t = boundary.otherMedium.get_refraction_index().x;
	u64 probLayerA = math::percentage_of(std::numeric_limits<u64>::max() - 1, 0.5f);
	bool reflect = rndSet.i0 < probLayerA;

	// Sample and get the pdfs of the second layer.
	math::PathSample sampleVal;
	math::BidirSampleValue otherVal;
	if(reflect) {
		rndSet.i0 = math::rescale_sample(rndSet.i0, 0, probLayerA-1);
		sampleVal = sample(params.a, incidentTS, boundary, rndSet, adjoint);
		if(sampleVal.pdf.forw.is_zero()) return sampleVal; // Discard
		otherVal = evaluate(params.b, incidentTS, sampleVal.excident, boundary);
	} else {
		rndSet.i0 = math::rescale_sample(rndSet.i0, probLayerA, std::numeric_limits<u64>::max());
		sampleVal = sample(params.b, incidentTS, boundary, rndSet, adjoint);
		if(sampleVal.pdf.forw.is_zero()) return sampleVal; // Discard
		otherVal = evaluate(params.a, incidentTS, sampleVal.excident, boundary);
	}
	const float iDotHabs = ei::abs(dot(incidentTS, boundary.get_halfTS(incidentTS, sampleVal.excident)));
	const float f = fresnel_dielectric(n_i, n_t, iDotHabs).f;
	float scaleS = float(sampleVal.pdf.forw);
	float scaleE = ei::abs(sampleVal.excident.z);
	if(reflect) { scaleS *= f; scaleE *= 1.0f - f; }
	else { scaleS *= 1.0f - f; scaleE *= f; }

	// Blend values and pdfs.
	sampleVal.pdf.forw = AngularPdf{ (float(otherVal.pdf.forw) + float(sampleVal.pdf.forw)) * 0.5f };
	sampleVal.pdf.back = AngularPdf{ (float(otherVal.pdf.back) + float(sampleVal.pdf.back)) * 0.5f };
	sampleVal.throughput = (sampleVal.throughput * scaleS + otherVal.value * scaleE)
						  / float(sampleVal.pdf.forw);
	return sampleVal;
}

// The evaluation routine
template<class LayerASample, class LayerBSample>
CUDA_FUNCTION math::BidirSampleValue evaluate(const MatSampleBlendFresnel<LayerASample, LayerBSample>& params,
											  const Direction& incidentTS,
											  const Direction& excidentTS,
											  Boundary& boundary) {
	// Evaluate both sub-layers
	auto valA = evaluate(params.a, incidentTS, excidentTS, boundary);
	auto valB = evaluate(params.b, incidentTS, excidentTS, boundary);
	// Determine the probability from Fresnel. It differs for both directions due to the sampling algorithm.
	float n_i = boundary.incidentMedium.get_refraction_index().x;
	float n_t = boundary.otherMedium.get_refraction_index().x;
	float iDotHabs = ei::abs(dot(incidentTS, boundary.get_halfTS(incidentTS, excidentTS)));
	float f = fresnel_dielectric(n_i, n_t, iDotHabs).f;
	// Blend their results
	valA.value = ei::lerp(valB.value, valA.value, f);
	valA.pdf.forw = AngularPdf{ (float(valB.pdf.forw) + float(valA.pdf.forw)) * 0.5f };
	valA.pdf.back = AngularPdf{ (float(valB.pdf.back) + float(valA.pdf.back)) * 0.5f };
	return valA;
}

// The albedo routine
template<class LayerASample, class LayerBSample>
CUDA_FUNCTION Spectrum albedo(const MatSampleBlendFresnel<LayerASample, LayerBSample>& params) {
	// TODO: some better approximation (I think the true albedo can only be found over
	// numeric integration).
	return albedo(params.a) * 0.5f
		 + albedo(params.b) * 0.5f;
}

template<class LayerASample, class LayerBSample>
CUDA_FUNCTION math::SampleValue emission(const MatSampleBlendFresnel<LayerASample, LayerBSample>& params, const scene::Direction& geoN, const scene::Direction& excident) {
	// Evaluate both sub-layers
	auto valA = emission(params.a, geoN, excident);
	auto valB = emission(params.b, geoN, excident);
	// Assume that at most one layer is emissive
	mAssert(valA.pdf.is_zero() || valB.pdf.is_zero());
	// TODO: blending and (blended) sampling of emissive models (independent of the other reflection properties)
	float cosIAbs = ei::abs(dot(geoN, excident));
	// TODO: needs media for correct evaluation
	float n_i = 1.0f; float n_t = 1.3f;
	float f = fresnel_dielectric(n_i, n_t, cosIAbs).f;
	valA.value = ei::lerp(valB.value, valA.value, f);
	valA.pdf += valB.pdf;
	return valA;
}

template<class LayerASample, class LayerBSample>
CUDA_FUNCTION float pdf_max(const MatSampleBlendFresnel<LayerASample, LayerBSample>& params) {
	// TODO: p based blending as above?
	return ei::max(pdf_max(params.a), pdf_max(params.b));
}

template MaterialSampleConcept<MatSampleBlendFresnel<MatSampleLambert, MatSampleLambert>>;
template MaterialConcept<MatBlendFresnel<MatLambert, MatLambert>>;

}}} // namespace mufflon::scene::materials
