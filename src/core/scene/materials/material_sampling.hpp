#pragma once

#include "material.hpp"
#include "medium.hpp"
#include "core/export/api.h"
#include "core/scene/types.hpp"
#include "util/log.hpp"
#include "util/assert.hpp"
#include "microfacet_base.hpp"
#include "lambert.hpp"
#include "emissive.hpp"
#include "blend.hpp"
#include "microfacet_specular.hpp"
#include <cuda_runtime.h>

namespace mufflon { namespace scene { namespace materials {

/*
 * Get the instanciated parameters for the evaluation of the material.
 * A parameter pack consists of the material type (see Materials) followed
 * by the two media handles and a specific parameters used in the
 * sampling/evaluation routines.
 * uvCoordinate: surface texture coordinate for fetching the textures.
 * outBuffer: pointer to a writeable buffer with at least get
 *		get_parameter_pack_size(device) memory.
 * Returns the size of the fetched data.
 */
CUDA_FUNCTION int fetch_subparam(Materials type, const char* subDesc, const UvCoordinate& uvCoordinate, char* subParam) {
	switch(type) {
		case Materials::LAMBERT: return as<LambertDesc<CURRENT_DEV>>(subDesc)->fetch(uvCoordinate, subParam);
		case Materials::EMISSIVE: return as<EmissiveDesc<CURRENT_DEV>>(subDesc)->fetch(uvCoordinate, subParam);
		case Materials::BLEND: return as<BlendDesc>(subDesc)->fetch(uvCoordinate, subParam);
		case Materials::TORRANCE: return as<TorranceDesc<CURRENT_DEV>>(subDesc)->fetch(uvCoordinate, subParam);
		default:
			mAssertMsg(false, "Material not (fully) implemented!");
	}
	return 0;
}

CUDA_FUNCTION int fetch(const MaterialDescriptorBase& desc, const UvCoordinate& uvCoordinate, ParameterPack* outBuffer) {
	outBuffer->type = desc.type;
	outBuffer->flags = desc.flags;
	outBuffer->innerMedium = desc.innerMedium;
	outBuffer->outerMedium = desc.outerMedium;
	const char* subDesc = as<char>(&desc) + sizeof(MaterialDescriptorBase);
	char* subParam = as<char>(outBuffer) + sizeof(ParameterPack);
	return fetch_subparam(desc.type, subDesc, uvCoordinate, subParam) + sizeof(ParameterPack);
}


// Used for regularization to avoid fireflies based on shading normal correction
constexpr float SHADING_NORMAL_EPS = 1e-3f;

// TODO: put the solve method into epsilon
CUDA_FUNCTION ei::Vec2 solve(const ei::Mat2x2 & _A, const ei::Vec2 & _b) {
	float detM = _A.m00 * _A.m11 - _A.m10 * _A.m01;
	float detX = _b.x   * _A.m11 - _b.y   * _A.m01;
	float detY = _A.m00 * _b.y   - _A.m10 * _b.x;
	return {detX / detM, detY / detM};
}



/*
 * Importance sampling of a generic material. This method switches to the specific
 * sampling routines internal.
 * incident: normalized incident direction. Points towards the surface.
 * adjoint: false if this is a view sub-path, true if it is a light sub-path.
 * boundary: input of the two media incident and opposite (not excident!).
 *		A halfvector based method should supply the sampled half vector
 *		via set to improve performance.
 */
// Kernel to split the sampling to specific implementations
CUDA_FUNCTION math::PathSample
sample_subdesc(Materials type,
			   const char* subParams,
			   const Direction& incidentTS,
			   Boundary& boundary,
			   const math::RndSet2_1& rndSet,
			   bool adjoint)
{
	switch(type)
	{
		case Materials::LAMBERT:
			return lambert_sample(*as<LambertParameterPack>(subParams), incidentTS, rndSet);
		case Materials::EMISSIVE:	// Not sampleable - simply let 'res' be the default value
			return math::PathSample{};
		case Materials::BLEND:
			return blend_sample(*as<BlendParameterPack>(subParams), incidentTS, boundary, rndSet, adjoint);
		case Materials::TORRANCE:
			return torrance_sample(*as<TorranceParameterPack>(subParams), incidentTS, boundary, rndSet);
		default: ;
#ifndef __CUDA_ARCH__
			logWarning("[materials::sample] Trying to evaluate unimplemented material type ", type);
#endif
			return math::PathSample{};
	}
}

CUDA_FUNCTION math::PathSample
sample(const TangentSpace& tangentSpace,
		const ParameterPack& params,
		const Direction& incident,
		const Medium* media,
		const math::RndSet2_1& rndSet,
		bool adjoint
) {
	// Cancel the path if shadowed shading normal (incident)
	float iDotG = -dot(incident, tangentSpace.geoN);
	float iDotN = -dot(incident, tangentSpace.shadingN);
	if(iDotG * iDotN <= 0.0f) return math::PathSample{};

	// Complete to-tangent space transformation.
	Direction incidentTS {
		-dot(incident, tangentSpace.shadingTX),
		-dot(incident, tangentSpace.shadingTY),
		iDotN
	};
	mAssert(ei::approx(len(incidentTS), 1.0f));

	Boundary boundary { media[params.get_medium(incidentTS.z)], media[params.get_medium(-incidentTS.z)] };

	// Use model specific subroutine
	const char* subParams = as<char>(&params) + sizeof(ParameterPack);
	math::PathSample res = sample_subdesc(params.type, subParams, incidentTS, boundary, rndSet, adjoint);

	// Early out if result is discarded anyway
	if(res.throughput == 0.0f) return res;

	// Transform local sample direction into global one.
	Direction globalDir = res.excident.x * tangentSpace.shadingTX
						+ res.excident.y * tangentSpace.shadingTY
						+ res.excident.z * tangentSpace.shadingN;
	mAssert(ei::approx(len(globalDir), 1.0f));

	// Cancel the path if shadowed shading normal (excident)
	float eDotG = dot(globalDir, tangentSpace.geoN);
	float eDotN = res.excident.z;
	if(eDotG * eDotN <= 0.0f) res.throughput = Spectrum{0.0f};
	res.excident = globalDir;

	// Shading normal correction
	if(adjoint) {
		res.throughput *= (SHADING_NORMAL_EPS + ei::abs(iDotN * eDotG))
						/ (SHADING_NORMAL_EPS + ei::abs(iDotG * eDotN));
	}

	mAssert(!isnan(res.throughput.x) && !isnan(res.excident.x) && !isnan(float(res.pdfF)) && !isnan(float(res.pdfB)));

	return res;
}



/*
 * Evaluate an BxDF and its associtated PDFs for two directions.
 * incident: normalized incident direction. Points towards the surface.
 * excident: normalized excident direction. Points away from the surface.
 * adjoint: false if the incident is a view sub-path, true if it is a light sub-path.
 * merge: Used to apply a different shading normal correction.
 */

// Kernel to call the specific evaluation routines
CUDA_FUNCTION math::EvalValue
evaluate_subdesc(Materials type,
				 const char* subParams,
				 const Direction& incidentTS,
				 const Direction& excidentTS,
				 Boundary& boundary,
				 bool adjoint,
				 bool merge) {
	switch(type)
	{
		case Materials::LAMBERT:
			return lambert_evaluate(*as<LambertParameterPack>(subParams), incidentTS, excidentTS);
		case Materials::EMISSIVE:
			return math::EvalValue{Spectrum{0.0f}, ei::abs(excidentTS.z), AngularPdf{0.0f}, AngularPdf{0.0f}};
		case Materials::BLEND:
			return blend_evaluate(*as<BlendParameterPack>(subParams), incidentTS, excidentTS, boundary, adjoint, merge);
		case Materials::TORRANCE:
			return torrance_evaluate(*as<TorranceParameterPack>(subParams), incidentTS, excidentTS, boundary);
		default:
#ifndef __CUDA_ARCH__
			logWarning("[materials::evaluate] Trying to evaluate unimplemented material type ", type);
#endif
			return math::EvalValue{};
	}
}

CUDA_FUNCTION math::EvalValue
evaluate(const TangentSpace& tangentSpace,
		 const ParameterPack& params,
		 const Direction& incident,
		 const Direction& excident,
		 const Medium* media,
		 bool adjoint,
		 bool merge
) {
	float iDotN = -dot(incident, tangentSpace.shadingN);
	float eDotN =  dot(excident, tangentSpace.shadingN);
	// Early out if possible
	if(!params.flags.is_set(MaterialPropertyFlags::REFLECTIVE) && iDotN * eDotN > 0.0f)
		return math::EvalValue{};
	if(!params.flags.is_set(MaterialPropertyFlags::REFRACTIVE) && iDotN * eDotN < 0.0f)
		return math::EvalValue{};

	float iDotG = -dot(incident, tangentSpace.geoN);
	float eDotG =  dot(excident, tangentSpace.geoN);
	// Shadow masking for the shading normal
	if(eDotG * eDotN <= 0.0f || iDotG * iDotN <= 0.0f)
		return math::EvalValue{};

	// Complete to-tangent space transformation.
	Direction incidentTS(
		-dot(incident, tangentSpace.shadingTX),
		-dot(incident, tangentSpace.shadingTY),
		iDotN
	);
	Direction excidentTS(
		dot(excident, tangentSpace.shadingTX),
		dot(excident, tangentSpace.shadingTY),
		eDotN
	);
	Boundary boundary { media[params.get_medium(incidentTS.z)], media[params.get_medium(-incidentTS.z)] };
	if(params.flags.is_set(MaterialPropertyFlags::HALFVECTOR_BASED)) {
		// Precompute the half vector (reduces divergence and instruction dependency later)
		boundary.get_halfTS(incidentTS, excidentTS);
	}

	// Call material implementation
	const char* subParams = as<char>(&params) + sizeof(ParameterPack);
	math::EvalValue res = evaluate_subdesc(params.type, subParams, incidentTS, excidentTS, boundary, adjoint, merge);

	// Early out if result is discarded anyway
	if(res.value == 0.0f) return res;
	mAssert(!isnan(res.value.x) && !isnan(float(res.pdfF)) && !isnan(float(res.pdfB)) && !isnan(res.cosOut));

	// Shading normal caused density correction.
	if(merge) {
		if(adjoint)
			res.value *= (SHADING_NORMAL_EPS + ei::abs(iDotN))
					   / (SHADING_NORMAL_EPS + ei::abs(iDotG));
		else
			res.value *= (SHADING_NORMAL_EPS + ei::abs(eDotN))
					   / (SHADING_NORMAL_EPS + ei::abs(eDotG));
	} else if(adjoint) {
		res.value *= (SHADING_NORMAL_EPS + ei::abs(iDotN * eDotG))
				   / (SHADING_NORMAL_EPS + ei::abs(iDotG * eDotN));
	}

	return res;
}

/*
 * Get the average color of the material (integral over all view direction in
 * a white furnace environment. Not necessarily the correct value - approximations
 * suffice.
 */
CUDA_FUNCTION Spectrum
albedo(Materials type, const char* subParams) {
	switch(type)
	{
		case Materials::LAMBERT: return lambert_albedo(*as<LambertParameterPack>(subParams));
		case Materials::EMISSIVE: return emissive_albedo(*as<EmissiveParameterPack>(subParams));
		case Materials::BLEND: return blend_albedo(*as<BlendParameterPack>(subParams));
		case Materials::TORRANCE: return torrance_albedo(*as<TorranceParameterPack>(subParams));
		default:
#ifndef __CUDA_ARCH__
			logWarning("[materials::albedo] Trying to evaluate unimplemented material type ", type);
#endif
			return Spectrum{0.0f};
	}
}
CUDA_FUNCTION Spectrum
albedo(const ParameterPack& params) {
	const char* subParams = as<char>(&params) + sizeof(ParameterPack);
	return albedo(params.type, subParams);
}

/*
 * Get the self emission into some direction.
 */
CUDA_FUNCTION Spectrum
emission(Materials type, const char* subParams, const scene::Direction& geoN, const scene::Direction& excident) {
	if(type == Materials::EMISSIVE)
		return dot(geoN, excident) > 0.0f ? as<EmissiveParameterPack>(subParams)->radiance : Spectrum{0.0f};
	if(type == Materials::BLEND)
		return blend_emission(*as<BlendParameterPack>(subParams), geoN, excident);
	// TODO: fresnel
	// Emission is not implemented in the majority of materials -> no check/redundant implementation
	return Spectrum{0.0f};
}

CUDA_FUNCTION Spectrum
emission(const ParameterPack& params, const scene::Direction& geoN, const scene::Direction& excident) {
	return emission(params.type, as<char>(&params) + sizeof(ParameterPack), geoN, excident);
}


/*
 * Get the size in bytes which are consumed for the complete parameter pack
 */
/*CUDA_FUNCTION int get_size(const ParameterPack& params) {
	switch(params.type)
	{
		case Materials::LAMBERT: return sizeof(LambertParameterPack) + sizeof(ParameterPack);
		case Materials::EMISSIVE: return sizeof(EmissiveParameterPack) + sizeof(ParameterPack);
		case Materials::BLEND: {
			//const char* layerA = as<char>(&params + 1);
			//const char* layerB = as<char>(&params) + params.offsetB;
			return sizeof(BlendParameterPack) + sizeof(ParameterPack);
				//+ get_size(
			// TODO: recursion or remove this entire method
		}
		default:
#ifndef __CUDA_ARCH__
			logWarning("[materials::get_size] Trying to evaluate unimplemented material type ", params.type);
#endif
			return 0;
	}
}*/

// Would be necessary for regularization
//virtual Spectrum get_maximum() const = 0;

}}} // namespace mufflon::scene::materials
