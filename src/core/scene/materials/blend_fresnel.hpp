#pragma once

#include "core/export/api.h"
#include "core/math/sampling.hpp"
#include "material_definitions.hpp"

namespace mufflon { namespace scene { namespace materials {

struct Refraction {
	float f;
	float cosTAbs;
};

// Dielectric Fresnel for unpolarized light
// etaSq: (n_i / n_t)^2
CUDA_FUNCTION Refraction fresnel_dielectric(float n_i, float n_t, float etaSq, float cosIAbs) {
	float cosTAbs = sqrt(ei::max(0.0f, 1.0f - etaSq * (1.0f - cosIAbs * cosIAbs)));
	float rParl = sdiv(n_t * cosIAbs - n_i * cosTAbs, n_t * cosIAbs + n_i * cosTAbs);
	float rPerp = sdiv(n_t * cosTAbs - n_i * cosIAbs, n_t * cosTAbs + n_i * cosIAbs);
	return {0.5f * (rParl * rParl + rPerp * rPerp), cosTAbs};
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
	float etaSq = ei::sq(n_i / n_t);
	float f = fresnel_dielectric(n_i, n_t, etaSq, ei::abs(incidentTS.z)).f;

//	float fa = params.factorA * sum(albedo(params.a));
//	float fb = params.factorB * sum(albedo(params.b));
//	float p = fa / (fa + fb);
	u64 probLayerA = math::percentage_of(std::numeric_limits<u64>::max() - 1, f);

	// Sample and get the pdfs of the second layer.
	math::PathSample sampleVal;
	math::BidirSampleValue otherVal;
	float scaleS, scaleE;
	if(rndSet.i0 < probLayerA) {
		rndSet.i0 = math::rescale_sample(rndSet.i0, 0, probLayerA-1);
		sampleVal = sample(params.a, incidentTS, boundary, rndSet, adjoint);
		if(sampleVal.pdf.forw.is_zero()) return sampleVal; // Discard
		otherVal = evaluate(params.b, incidentTS, sampleVal.excident, boundary);
		scaleS = float(sampleVal.pdf.forw) * f;
		scaleE = 1.0f - f;
	} else {
		rndSet.i0 = math::rescale_sample(rndSet.i0, probLayerA, std::numeric_limits<u64>::max());
		sampleVal = sample(params.b, incidentTS, boundary, rndSet, adjoint);
		if(sampleVal.pdf.forw.is_zero()) return sampleVal; // Discard
		otherVal = evaluate(params.a, incidentTS, sampleVal.excident, boundary);
		scaleS = float(sampleVal.pdf.forw) * (1.0f - f);
		scaleE = f;
		f = 1.0f - f;
	}
	scaleE *= ei::abs(sampleVal.excident.z);
	// The Fresnel probability in the backward direction is different than that from
	// forward (term depents on the selected direction).
	if(incidentTS.z * sampleVal.excident.z < 0) {
		float t = n_i; n_i = n_t; n_t = t;
		etaSq = 1.0f / etaSq;
	}
	float fB = fresnel_dielectric(n_i, n_t, etaSq, ei::abs(sampleVal.excident.z)).f;

	// Blend values and pdfs.
	float origPdf = float(sampleVal.pdf.forw);
	sampleVal.pdf.forw = AngularPdf{ ei::lerp(float(otherVal.pdf.forw), float(sampleVal.pdf.forw), f) };
	sampleVal.pdf.back = AngularPdf{ ei::lerp(float(otherVal.pdf.back), float(sampleVal.pdf.back), fB) };
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
	float etaSq = ei::sq(n_i / n_t);
	float fF = fresnel_dielectric(n_i, n_t, etaSq, ei::abs(incidentTS.z)).f;
	if(incidentTS.z * excidentTS.z < 0) {
		float t = n_i; n_i = n_t; n_t = t;
		etaSq = 1.0f / etaSq;
	}
	float fB = fresnel_dielectric(n_i, n_t, etaSq, ei::abs(excidentTS.z)).f;
/*	float fa = params.factorA * sum(albedo(params.a));
	float fb = params.factorB * sum(albedo(params.b));
	float p = fa / (fa + fb);// TODO: precompute in fetch?*/
	// Blend their results
	valA.value = ei::lerp(valB.value, valA.value, fF);
	valA.pdf.forw = AngularPdf{ ei::lerp(float(valB.pdf.forw), float(valA.pdf.forw), fF) };
	valA.pdf.back = AngularPdf{ ei::lerp(float(valB.pdf.back), float(valA.pdf.back), fB) };
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
	float etaSq = ei::sq(n_i / n_t);
	float f = fresnel_dielectric(n_i, n_t, etaSq, cosIAbs).f;
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
