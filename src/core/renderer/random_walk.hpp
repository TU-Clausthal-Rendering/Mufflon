#pragma once

#include "path_util.hpp"
#include "core/scene/materials/material_sampling.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/accel_structs/intersection.hpp"

namespace mufflon { namespace renderer {

// Collection of parameters produced or used by a random walk
// TODO: vertex customization?
struct PathHead {
	Throughput throughput;			// General throughput with guide heuristics
	scene::Point position;
	AngularPdf pdfF;				// Forward PDF of the last sampling PDF
	scene::Direction excident;		// May be zero-vector for start points

	PathHead() {}
	PathHead(const cameras::RaySample& camSample) :
		throughput{ei::Vec3{camSample.w / float(camSample.pdf)}, 1.0f},
		position{camSample.origin},
		pdfF{camSample.pdf},
		excident{camSample.excident}
	{}
};

/*
 * The random walk routine is a basic primitive of most Monte-Carlo based renderers.
 * It is meant to be used as a subfunction of any renderer and summariezes effects of
 * sampling and roussian roulette.
 * This also computes the attenuation through the medium.
 *
 * excidentRay: position and excident direction of a completed sampling event
 * rayPdf: the angular PDF of this sampling event
 * rayContribution: throughput of the ray (usually BxDF / PDF)
 * u0: a uniform random number for roussion roulette.
 * throughput [in/out]: The throughput value which is changed by the current sampling event/russion roulette.
 * nextHit [out]: The intersection result of the walk (if any)
 * returns: true if there is a nextHit. false if the path is canceled/misses the scene.
 */
//template < typename VertexType >
CUDA_FUNCTION bool walk(const ei::Ray& excidentRay, AngularPdf rayPdf,
						const Spectrum& eventThroughput, const scene::materials::Medium& medium, float u0,
						Throughput& throughput, scene::accel_struct::RayIntersectionResult& nextHit
) {
	// Update throughputs
	throughput.weight *= eventThroughput;
	throughput.guideWeight *= 1.0f - expf(-(rayPdf * rayPdf) / 5.0f);

	// Russian roulette
	float continuationPropability = ei::min(max(eventThroughput) + 0.05f, 1.0f);
	if(u0 >= continuationPropability)	// The smaller the contribution the more likely the kill
		return false;
	else {
		// Continue and compensate if rouletteWeight < 1.
		throughput.weight /= continuationPropability;
		throughput.guideWeight /= continuationPropability;
	}

	// TODO: optional energy clamping

	// Go to the next intersection
	//bool didHit = first_hit(, nextHit);

	// Compute attenuation
	Spectrum transmission = medium.get_transmission( nextHit.hitT );
	throughput.weight *= transmission;
	throughput.guideWeight *= avg(transmission);

	return true;
}

}} // namespace mufflon::renderer