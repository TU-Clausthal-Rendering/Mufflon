#pragma once

#include "material_definitions.hpp"
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
#include "microfacet_refractive.hpp"
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
CUDA_FUNCTION int fetch(const MaterialDescriptorBase& desc, const UvCoordinate& uvCoordinate, ParameterPack* outBuffer) {
	auto MAT_TEX_COUNT = details::enumerate_tex_counts( std::make_integer_sequence<int, int(Materials::NUM)>{} );
	// 1. Fetch date in a unified style to reduce divergence
	const textures::ConstTextureDevHandle_t<CURRENT_DEV>* tex = as<textures::ConstTextureDevHandle_t<CURRENT_DEV>>(&desc + 1);
	ei::Vec4 texels[MAT_MAX_TEX_COUNT()];
	for(int i = 0; i < MAT_TEX_COUNT[int(desc.type)]; ++i)
		texels[i] = sample(tex[i], uvCoordinate);
	// 2. Convert the data into an output sample value.
	outBuffer->type = desc.type;
	outBuffer->flags = desc.flags;
	outBuffer->innerMedium = desc.innerMedium;
	outBuffer->outerMedium = desc.outerMedium;
	const char* subParams = as<char>(&desc + 1) + sizeof(textures::ConstTextureDevHandle_t<CURRENT_DEV>) * MAT_TEX_COUNT[int(desc.type)];
	material_switch(desc.type,
		*as<typename MatType::SampleType>(outBuffer+1) = fetch(tex, texels, 0, *as<typename MatType::NonTexParams>(subParams));
		return sizeof(MaterialDescriptorBase) + sizeof(MatType::SampleType);
	);
	return sizeof(MaterialDescriptorBase);
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
	math::PathSample res;
	material_switch(params.type,
		res = sample(*as<typename MatType::SampleType>(subParams), incidentTS, boundary, rndSet, adjoint);
		break;
	);

	// Early out if result is discarded anyway
	if(res.throughput == 0.0f) return res;

	// Transform local sample direction into global one.
	Direction globalDir = res.excident.x * tangentSpace.shadingTX
						+ res.excident.y * tangentSpace.shadingTY
						+ res.excident.z * tangentSpace.shadingN;
	mAssert(ei::approx(len(globalDir), 1.0f, 1e-4f));

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

	mAssert(!isnan(res.throughput.x) && !isnan(res.excident.x) && !isnan(float(res.pdf.forw)) && !isnan(float(res.pdf.back)));

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
	// Swap directions to guarentee symmetry between light and view path
	if(adjoint) {
		Direction tmp = incidentTS;
		incidentTS = excidentTS;
		excidentTS = tmp;
	}
	Boundary boundary { media[params.get_medium(incidentTS.z)], media[params.get_medium(-incidentTS.z)] };
	if(params.flags.is_set(MaterialPropertyFlags::HALFVECTOR_BASED)) {
		// Precompute the half vector (reduces divergence and instruction dependency later)
		boundary.get_halfTS(incidentTS, excidentTS);
	}

	// Call material implementation
	const char* subParams = as<char>(&params) + sizeof(ParameterPack);
	math::BidirSampleValue res;
	material_switch(params.type,
		res = evaluate(*as<typename MatType::SampleType>(subParams),
			incidentTS, excidentTS, boundary);
		break;
	);

	// Early out if result is discarded anyway
	if(res.value == 0.0f) return math::EvalValue{};
	mAssert(!isnan(res.value.x) && !isnan(float(res.pdf.forw)) && !isnan(float(res.pdf.back)));

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

	return math::EvalValue{
		res.value, ei::abs(eDotN),
		// Swap back output values if we swapped the directions before
		adjoint ? res.pdf.back : res.pdf.forw,
		adjoint ? res.pdf.forw : res.pdf.back
	};
}

/*
 * Get the average color of the material (integral over all view direction in
 * a white furnace environment. Not necessarily the correct value - approximations
 * suffice.
 */
CUDA_FUNCTION Spectrum
albedo(const ParameterPack& params) {
	const char* subParams = as<char>(&params) + sizeof(ParameterPack);
	material_switch(params.type,
		return albedo(*as<typename MatType::SampleType>(subParams));
	);
	return Spectrum{0.0f};
}

/*
 * Get the self emission into some direction.
 */
CUDA_FUNCTION math::SampleValue
emission(const ParameterPack& params, const scene::Direction& geoN, const scene::Direction& excident) {
	const char* subParams = as<char>(&params) + sizeof(ParameterPack);
	material_switch(params.type,
		return emission(*as<typename MatType::SampleType>(subParams), geoN, excident);
	);
	return math::SampleValue{};
}

}}} // namespace mufflon::scene::materials
