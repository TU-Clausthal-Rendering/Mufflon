#pragma once

#include "material.hpp"
#include "medium.hpp"
#include "core/export/api.h"
#include "core/scene/types.hpp"
#include "util/log.hpp"
#include "util/assert.hpp"
#include "lambert.hpp"
#include <cuda_runtime.h>

namespace mufflon { namespace scene { namespace materials {

/*
 * Get the instanciated parameters for the evaluation of the material.
 * A parameter pack consits of the material type (see Materials) followed
 * by the two media handles and and specific parameters used in the
 * sampling/evaluation routines.
 * uvCoordinate: surface texture coordinate for fetching the textures.
 * outBuffer: pointer to a writeable buffer with at least get
 *		get_parameter_pack_size(device) memory.
 */
CUDA_FUNCTION void fetch(const HandlePack& handles, const UvCoordinate& uvCoordinate, ParameterPack* outBuffer) {
	switch(handles.type) {
		case Materials::LAMBERT: as<LambertHandlePack<CURRENT_DEV>>(handles).fetch(uvCoordinate, outBuffer);
	}
	mAssertMsg(false, "Material not (fully) implemented!");
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
 */
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

	// Use model specific subroutine
	math::PathSample res;
	switch(params.type)
	{
		case Materials::LAMBERT: {
			res = lambert_sample(static_cast<const LambertParameterPack&>(params), incidentTS, rndSet);
		} break;
		default: ;
#ifndef __CUDA_ARCH__
			logWarning("[materials::sample] Trying to evaluate unimplemented material type ", params.type);
#endif
	}

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

	return res;
}

/*
 * Evaluate an BxDF and its associtated PDFs for two directions.
 * incident: normalized incident direction. Points towards the surface.
 * excident: normalized excident direction. Points away from the surface.
 * adjoint: false if the incident is a view sub-path, true if it is a light sub-path.
 * merge: Used to apply a different shading normal correction.
 */
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
	Direction halfTS { 0.0f, 0.0f, 1.0f };
	if(params.flags.is_set(MaterialPropertyFlags::HALFVECTOR_BASED)) {
		const Medium& inMedium = media[params.get_medium(incidentTS.z)];
		const Medium& exMedium = media[params.get_medium(excidentTS.z)];
		halfTS = inMedium.get_refraction_index().x * incidentTS + exMedium.get_refraction_index().x * excidentTS;
		float l = len(halfTS) * ei::sgn(halfTS.z); // Half vector always on side of the normal
		halfTS = sdiv(halfTS, l);
	}

	// Call material implementation
	math::EvalValue res;
	switch(params.type)
	{
		case Materials::LAMBERT: {
			res = lambert_evaluate(static_cast<const LambertParameterPack&>(params), incidentTS, excidentTS);
		} break;
		default: ;
#ifndef __CUDA_ARCH__
			logWarning("[materials::evaluate] Trying to evaluate unimplemented material type ", params.type);
#endif
	}

	// Early out if result is discarded anyway
	if(res.value == 0.0f) return res;

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
albedo(const ParameterPack& params) {
	switch(params.type)
	{
		case Materials::LAMBERT: return lambert_albedo(static_cast<const LambertParameterPack&>(params));
		default:
#ifndef __CUDA_ARCH__
			logWarning("[materials::albedo] Trying to evaluate unimplemented material type ", params.type);
#endif
			return Spectrum{0.0f};
	}
}

/*
 * Get the self emission into some direction.
 */
CUDA_FUNCTION Spectrum
emission(const ParameterPack& params, const scene::Direction& excident) {
	switch(params.type)
	{
		case Materials::LAMBERT: return Spectrum{0.0f};
		default:
#ifndef __CUDA_ARCH__
			logWarning("[materials::emission] Trying to evaluate unimplemented material type ", params.type);
#endif
			return Spectrum{0.0f};
	}
}


/*
 * Get the size in bytes which are consumed for the complete parameter pack
 */
CUDA_FUNCTION int get_size(const ParameterPack& params) {
	switch(params.type)
	{
		case Materials::LAMBERT: return sizeof(LambertParameterPack);
			// Others might be recursive: write get_size for the specific materials
		default:
#ifndef __CUDA_ARCH__
			logWarning("[materials::get_size] Trying to evaluate unimplemented material type ", params.type);
#endif
			return 0;
	}
}

// Would be necessary for regularization
//virtual Spectrum get_maximum() const = 0;

}}} // namespace mufflon::scene::materials
