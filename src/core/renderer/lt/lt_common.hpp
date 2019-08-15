#pragma once

#include "lt_params.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"

namespace mufflon { namespace renderer {

// The pure connection light tracer does not need any additional vertex information
using LtPathVertex = PathVertex<VertexExtension>;


/*
 * Create one sample path (random walk of a photon) and connect each vertex to the camera
 */
CUDA_FUNCTION void lt_sample(typename LtTargets::template RenderBufferType<CURRENT_DEV> outputBuffer,
							 const scene::SceneDescriptor<CURRENT_DEV>& scene,
							 const LtParameters& params,
							 const int idx,
							 math::Rng& rng) {
	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	LtPathVertex vertex;
	VertexSample sample;
	// Create a start vertex for the path
	math::RndSet2 rndStart { rng.next() };
	u64 lightTreeSeed = rng.next();
	scene::lights::Emitter p = scene::lights::emit(scene, idx, outputBuffer.get_num_pixels(),
		lightTreeSeed, rndStart);
	LtPathVertex::create_light(&vertex, nullptr, p);

	// Create a camera vertex for easier back-projection
	LtPathVertex camera;
	LtPathVertex::create_camera(&camera, nullptr, scene.camera.get(), Pixel{0,0}, rng.next());

	int pathLen = 0;
	do {
		if(pathLen+1 >= params.minPathLength && pathLen+1 <= params.maxPathLength) {
			// Connect to the camera. An MIS is not necessary, because this connection is the
			// only event in this renderer
			auto connection = LtPathVertex::get_connection(camera, vertex);
			Pixel outPixel;
			math::EvalValue cval = camera.evaluate(connection.dir, scene.media, outPixel, false);
			if(outPixel.x != -1) {
				math::EvalValue lval = vertex.evaluate(-connection.dir, scene.media, outPixel, true);
				Spectrum bxdfProd = cval.value * lval.value;
				float cosProd = cval.cosOut * lval.cosOut;
				// Early out if there would not be a contribution (estimating the materials is usually
				// cheaper than the any-hit test).
				if(any(greater(bxdfProd, 0.0f)) && cosProd > 0.0f) {
					// Shadow test
					if(!scene::accel_struct::any_intersection(
						scene, connection.v0, vertex.get_position(connection.v0),
						camera.get_geometric_normal(), vertex.get_geometric_normal(), connection.dir)) {
						bxdfProd /= connection.distanceSq;

						outputBuffer.contribute<RadianceTarget>(outPixel, throughput.weight * cosProd * bxdfProd);
						outputBuffer.contribute<LightnessTarget>(outPixel, throughput.guideWeight * cosProd);
					}
				}
			}
		}

		// Walk
		math::RndSet2_1 rnd { rng.next(), rng.next() };
		float rndRoulette = math::sample_uniform(u32(rng.next()));
		if(walk(scene, vertex, rnd, rndRoulette, true, throughput, vertex, sample) != WalkResult::HIT)
			break;
		++pathLen;
	} while(pathLen < params.maxPathLength);
}

}} // namespace mufflon::renderer