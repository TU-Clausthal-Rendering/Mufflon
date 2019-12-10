#pragma once

#include "path_util.hpp"
#include "core/scene/materials/material_sampling.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/accel_structs/tangent_space.hpp"
#include <math.h>

namespace mufflon { namespace renderer {

enum class WalkResult {
	CANCEL,			// The walk is cancled due to sampling related issus (e.g. self hit or Russian roulette)
	BACKGROUND,		// The walk missed the scene, but created one last void vertex
	HIT,			// Reached a surface
};

/*
 * The random walk routine is a basic primitive of most Monte-Carlo based renderers.
 * It is meant to be used as a subfunction of any renderer and summariezes effects of
 * sampling and roussian roulette.
 * This also computes the attenuation through the medium.
 *
 * Args: additional optional arguments to pass information from the renderer
 * to the ext.update //and ext.init TODO// method.
 *
 * vertex: generic vertex which is sampled to proceed on the path
 * adjoint: is this a light sub-path?
 * rndSet: a set of random numbers to sample the vertex
 * u0: a uniform random number for roussion roulette or -1 to disable roulette.
 * throughput [in/out]: The throughput value which is changed by the current sampling event/russion roulette.
 * sampledDir [out]: the ray which was sampled in the walk
 * nextHit [out]: The intersection result of the walk (if any)
 * returns: true if there is a nextHit. false if the path is canceled/misses the scene.
 */
template < typename VertexType, typename... Args >
inline CUDA_FUNCTION WalkResult walk(const scene::SceneDescriptor<CURRENT_DEV>& scene,
							  const VertexType& vertex,
							  const math::RndSet2_1& rndSet, float u0,
							  bool adjoint,
							  Spectrum& throughput,
							  VertexType& outVertex,
							  VertexSample& outSample,
							  Args&&... args
) {
	// Sample the vertex's outgoing direction
	outSample = vertex.sample(scene.aabb, scene.media, rndSet, adjoint);
	if(outSample.type == math::PathEventType::INVALID) {
		throughput = Spectrum { 0.0f };
		return WalkResult::CANCEL;
	}
	mAssert(!isnan(outSample.excident.x) && !isnan(outSample.excident.y) && !isnan(outSample.excident.z)
		&& !isnan(outSample.origin.x) && !isnan(outSample.origin.y) && !isnan(outSample.origin.z)
		&& !isnan(float(outSample.pdf.forw)) && !isnan(float(outSample.pdf.back)));
	vertex.ext().update(vertex, outSample.excident, outSample, args...);

	// Update throughputs
	throughput *= outSample.throughput;

	// Russian roulette
	float continuationProbability = 1.0f;
	if(u0 >= 0.0f) {
		continuationProbability = ei::min(max(throughput) + 0.1f, 1.0f);
		if(u0 >= continuationProbability) {	// The smaller the contribution the more likely the kill
			throughput = Spectrum { 0.0f };
			return WalkResult::CANCEL;
		} else {
			// Continue and compensate if rouletteWeight < 1.
			throughput /= continuationProbability;
		}
	}

	// TODO: optional energy clamping

	// Go to the next intersection
	ei::Ray ray {outSample.origin, outSample.excident};
	scene::accel_struct::RayIntersectionResult nextHit =
		scene::accel_struct::first_intersection(scene, ray, vertex.get_geometric_normal(), scene::MAX_SCENE_SIZE);

	// Compute attenuation
	const scene::materials::Medium& currentMedium = scene.media[outSample.medium];
	Spectrum transmission = currentMedium.get_transmission( nextHit.distance );
	throughput *= transmission;
	mAssert(!isnan(throughput.x) && !isnan(throughput.y) && !isnan(throughput.z));

	Connection connection {
		ray.direction, ei::sq(nextHit.distance), ray.origin, nextHit.distance
	};

	// If we missed the scene, terminate the ray
	if(nextHit.hitId.instanceId < 0) {
		VertexType::create_void(&outVertex, &vertex, outSample.excident);
		outVertex.ext().update(vertex, outVertex, outSample.pdf, connection, throughput,
			continuationProbability, transmission, args...);
		return WalkResult::BACKGROUND;
	}

	// Create the new surface vertex
	ei::Vec3 position = ray.origin + outSample.excident * nextHit.distance;
	// Get tangent space and parameter pack from nextHit
	const scene::TangentSpace tangentSpace = scene::accel_struct::tangent_space_geom_to_shader(scene, nextHit);
	// Finalize
	VertexType::create_surface(&outVertex, &vertex, nextHit, scene.get_material(nextHit.hitId),
				position, tangentSpace, outSample.excident);
	outVertex.ext().update(vertex, outVertex, outSample.pdf, connection, throughput,
		continuationProbability, transmission, args...);
	return WalkResult::HIT;
}

}} // namespace mufflon::renderer
