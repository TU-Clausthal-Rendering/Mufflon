#pragma once

#include "material.hpp"
#include "core/scene/types.hpp"
#include <cuda_runtime.h>

namespace mufflon { namespace scene { namespace material {

	/*
	 * A RndSet is a fixed size set of random numbers which may be consumed by a material
	 * sampler. The first two values are standard uniform distributed floating point samples
	 * which should be used to sample the direction.
	 * The third value is open to additional requirements as layer decisions.
	 */
	struct RndSet {
		float x0;	// In [0,1)
		float x1;	// In [0,1)
		u32 i0;		// Full 32 bit random information
	};

	// Return value of an importance sampler
	struct Sample {
		Spectrum throughput {1.0f};		// BxDF * cosθ / pdfF
		Direction excident {0.0f};		// The sampled direction
		float pdfF {0.0f};				// Sampling PDF in forward direction (current sampler)
		float pdfB {0.0f};				// Sampling PDF with reversed incident and excident directions
		enum class Type: u32 {			// Type of interaction
			INVALID,
			REFLECTED,
			REFRACTED,
		} type = Type::INVALID;
	};

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
		   bool adjoint);

	// Return value of a BxDF evaluation function
	struct EvalValue {
		Spectrum bxdf;			// BRDF or BTDF value
		float cosThetaOut;		// Outgoing cosine
		float pdfF;				// Sampling PDF in forward direction 
		float pdfB;				// Sampling PDF with reversed incident and excident directions
	};

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
			 bool merge);

}}} // namespace mufflon::scene::material