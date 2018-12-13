#pragma once

#include "path_util.hpp"
#include "core/scene/materials/material_sampling.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/descriptors.hpp"

namespace mufflon { namespace renderer {

// Collection of parameters produced or used by a random walk
// TODO: vertex customization?
struct PathHead {
	Throughput throughput;			// General throughput with guide heuristics
	scene::Point position;
	AngularPdf pdfF;				// Forward PDF of the last sampling PDF
	scene::Direction excident;		// May be zero-vector for start points

	PathHead() {}
/*	PathHead(const cameras::Importon& camSample) :
		throughput{ei::Vec3{camSample.w / float(camSample.pdf)}, 1.0f},
		position{camSample.origin},
		pdfF{camSample.pdf},
		excident{camSample.excident}
	{}*/
};

/*
 * The random walk routine is a basic primitive of most Monte-Carlo based renderers.
 * It is meant to be used as a subfunction of any renderer and summariezes effects of
 * sampling and roussian roulette.
 * This also computes the attenuation through the medium.
 *
 * vertex: generic vertex which is sampled to proceed on the path
 * adjoint: is this a light sub-path?
 * rndSet: a set of random numbers to sample the vertex
 * u0: a uniform random number for roussion roulette.
 * throughput [in/out]: The throughput value which is changed by the current sampling event/russion roulette.
 * sampledDir [out]: the ray which was sampled in the walk
 * nextHit [out]: The intersection result of the walk (if any)
 * returns: true if there is a nextHit. false if the path is canceled/misses the scene.
 */
template < typename VertexType >
CUDA_FUNCTION bool walk(const scene::SceneDescriptor<CURRENT_DEV>& scene,
						const VertexType& vertex,
						const math::RndSet2_1& rndSet, float u0,
						bool adjoint,
						Throughput& throughput,
						VertexType* outVertex
) {
	// Sample the vertex's outgoing direction
	VertexSample sample = vertex.sample(scene.media, rndSet, adjoint);
	if(sample.type == math::PathEventType::INVALID) return false;

	// Update throughputs
	throughput.weight *= sample.throughput;
	throughput.guideWeight *= 1.0f - expf(-(sample.pdfF * sample.pdfF) / 5.0f);

	// Russian roulette
	float continuationPropability = ei::min(max(sample.throughput) + 0.05f, 1.0f);
	if(u0 >= continuationPropability)	// The smaller the contribution the more likely the kill
		return false;
	else {
		// Continue and compensate if rouletteWeight < 1.
		throughput.weight /= continuationPropability;
		throughput.guideWeight /= continuationPropability;
	}

	// TODO: optional energy clamping

	// Go to the next intersection
	scene::accel_struct::RayIntersectionResult<CURRENT_DEV> nextHit;
	//ei::Ray ray {sample.origin, sample.excident};
	//bool didHit = first_hit(, nextHit);

	// Compute attenuation
	const scene::materials::Medium& currentMedium = scene.media[sample.medium];
	Spectrum transmission = currentMedium.get_transmission( nextHit.hitT );
	throughput.weight *= transmission;
	throughput.guideWeight *= avg(transmission);

	// if(!didHit) return false;

	// Create the new surface vertex
	ei::Vec3 position = vertex.get_position() + sample.excident * nextHit.hitT;
	float incidentCos = dot(nextHit.normal, sample.excident); // TODO: shading normal?
	// TODO: get tangent space and parameter pack from nextHit
	scene::MaterialIndex matIdx = 0; // TODO
	char materialBuffer[scene::materials::MAX_MATERIAL_PARAMETER_SIZE];
	scene::materials::fetch(scene.get_material(matIdx), scene::UvCoordinate{}, as<scene::materials::ParameterPack>(materialBuffer));
	VertexType::create_surface(outVertex, &vertex, *as<scene::materials::ParameterPack>(materialBuffer),
				{ position, sample.pdfF.to_area_pdf(incidentCos,ei::sq(nextHit.hitT)) },
				scene::TangentSpace{}, sample.excident);

	return true;
}

}} // namespace mufflon::renderer