#pragma once

#include "material.hpp"
#include "core/scene/types.hpp"
#include "util/log.hpp"
#include "lambert.hpp"
#include <cuda_runtime.h>

namespace mufflon { namespace scene { namespace material {

	/*
	 * Importance sampling of a generic material. This method switches to the specific
	 * sampling routines internal.
	 * incident: normalized incident direction. Points towards the surface.
	 * adjoint: false if this is a view sub-path, true if it is a light sub-path.
	 */
	__host__ __device__ Sample
	sample(const TangentSpace& tangentSpace,
		   const ParameterPack& params,
		   const Direction& incident,
		   const RndSet& rndSet,
		   bool adjoint) {
		// Cancel the path if shadowed shading normal (incident)
		float iDotG = -dot(incident, tangentSpace.geoN);
		float iDotN = -dot(incident, tangentSpace.shadingN);
		if(iDotG * iDotN <= 0.0f) return Sample{};

		// Complete to-tangent space transformation.
		Direction incidentTS(
			-dot(incident, tangentSpace.tU),
			-dot(incident, tangentSpace.tV),
			iDotN
		);

		// Use model specific subroutine
		Sample res;
		switch(params.type)
		{
			case Materials::LAMBERT: {
				res = lambert_sample(static_cast<const LambertParameterPack&>(params), incidentTS, rndSet);
			} break;
			default: ;
#ifndef __CUDA_ARCH__
				logWarning("[material::sample] Trying to evaluate unimplemented material type ", params.type);
#endif
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
	__host__ __device__ EvalValue
	evaluate(const TangentSpace& tangentSpace,
			 const ParameterPack& params,
			 const Direction& incident,
			 const Direction& excident,
			 bool adjoint,
			 bool merge) {
		return EvalValue{};
	}

	/*
	 * Get the average color of the material (integral over all view direction in
	 * a white furnace environment. Not necessarily the correct value - approximations
	 * suffice.
	 */
	__host__ __device__ Spectrum
	albedo(const ParameterPack& params) {
		switch(params.type)
		{
			case Materials::LAMBERT: return lambert_albedo(static_cast<const LambertParameterPack&>(params));
			default:
#ifndef __CUDA_ARCH__
				logWarning("[material::albedo] Trying to evaluate unimplemented material type ", params.type);
#endif
				return Spectrum{0.0f};
		}
	}

	// Would be necessary for regularization
	//virtual Spectrum get_maximum() const = 0;
}

}}} // namespace mufflon::scene::material