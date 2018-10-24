#include "sample.hpp"
#include "lambert.hpp"
#include "util/log.hpp"

namespace mufflon { namespace scene { namespace material {

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
			logWarning("Trying to evaluate unimplemented material type ", params.type);
#endif
	}

	return res;
}


__host__ __device__ EvalValue
evaluate(const TangentSpace& tangentSpace,
		 const ParameterPack& params,
		 const Direction& incident,
		 const Direction& excident,
		 bool adjoint,
		 bool merge) {
	return EvalValue{};
}

}}} // namespace mufflon::scene::material