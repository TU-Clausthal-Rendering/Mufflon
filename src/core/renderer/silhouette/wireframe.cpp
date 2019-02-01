#include "wireframe.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/scene/scene.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include <random>

namespace mufflon::renderer::silhouette {

using PtPathVertex = PathVertex<u8, 4>;

WireframeRenderer::WireframeRenderer() {
	// TODO: init one RNG per thread?
	m_rngs.emplace_back(static_cast<u32>(std::random_device()()));

	// The PT does not need additional memory resources like photon maps.
}

void WireframeRenderer::iterate(OutputHandler& outputBuffer) {
	// (Re) create the random number generators
	if(m_rngs.size() != outputBuffer.get_num_pixels()
	   || m_reset)
		init_rngs(outputBuffer.get_num_pixels());

	RenderBuffer<Device::CPU> buffer = outputBuffer.begin_iteration<Device::CPU>(m_reset);
	if(m_reset) {
		// TODO: reset output buffer
		// Reacquire scene descriptor (partially?)
		m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, outputBuffer.get_resolution());
	}
	m_reset = false;

	// TODO: call sample in a parallel way for each output pixel
	// TODO: better pixel order?
	// TODO: different scheduling?
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < outputBuffer.get_num_pixels(); ++pixel) {
		this->sample(Pixel{ pixel % outputBuffer.get_width(), pixel / outputBuffer.get_width() }, buffer, m_sceneDesc);
	}

	outputBuffer.end_iteration<Device::CPU>();
}

void WireframeRenderer::reset() {
	this->m_reset = true;
}

void WireframeRenderer::sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer,
						   const scene::SceneDescriptor<Device::CPU>& scene) {
	int pixel = coord.x + coord.y * outputBuffer.get_width();

	//m_params.maxPathLength = 2;

	Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	u8 vertexBuffer[256]; // TODO: depends on materials::MAX_MATERIAL_PARAMETER_SIZE
	PtPathVertex* vertex = as<PtPathVertex>(vertexBuffer);
	// Create a start for the path
	int s = PtPathVertex::create_camera(vertex, vertex, scene.camera.get(), coord, m_rngs[pixel].next());
	mAssertMsg(s < 256, "vertexBuffer overflow.");

	math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
	VertexSample sample = vertex->sample(scene.media, rnd, false);
	ei::Ray ray{ sample.origin, sample.excident };

	while(true) {
		scene::accel_struct::RayIntersectionResult nextHit =
			scene::accel_struct::first_intersection_scene_lbvh<CURRENT_DEV>(scene, ray, vertex->get_primitive_id(), scene::MAX_SCENE_SIZE);
		if(nextHit.hitId.instanceId < 0) {
			auto background = evaluate_background(scene.lightTree.background, ray.direction);
			if(any(greater(background.value, 0.0f))) {
				outputBuffer.contribute(coord, throughput, background.value,
										ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
										ei::Vec3{ 0, 0, 0 });
			}
			break;
		} else {
			const scene::LodDescriptor<CURRENT_DEV>& object = scene.lods[scene.lodIndices[nextHit.hitId.instanceId]];

			// TODO: remove, it's for wireframe rendering
			if(static_cast<u32>(nextHit.hitId.primId) < object.polygon.numTriangles) {
				float minBary = nextHit.surfaceParams.barycentric.x;
				minBary = minBary > nextHit.surfaceParams.barycentric.y ?
					nextHit.surfaceParams.barycentric.y : minBary;
				const float baryZ = 1.f - nextHit.surfaceParams.barycentric.x - nextHit.surfaceParams.barycentric.y;
				minBary = minBary > baryZ ? baryZ : minBary;

				if(minBary > m_params.thickness) {
					ray.origin = ray.origin + ray.direction * (nextHit.hitT + 0.0001f);
				} else {
					outputBuffer.contribute(coord, throughput, ei::Vec3{ 1.f },
											ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
											ei::Vec3{ 0, 0, 0 });
					break;
				}
			} else if(static_cast<u32>(nextHit.hitId.primId) < object.polygon.numTriangles + object.polygon.numQuads) {
				if((nextHit.surfaceParams.bilinear.x > m_params.thickness && nextHit.surfaceParams.bilinear.x < 1.f - m_params.thickness)
				   && (nextHit.surfaceParams.bilinear.y > m_params.thickness && nextHit.surfaceParams.bilinear.y < 1.f - m_params.thickness)) {
					ray.origin = ray.origin + ray.direction * (nextHit.hitT + 0.0001f);
				} else {
					outputBuffer.contribute(coord, throughput, ei::Vec3{ 1.f },
											ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
											ei::Vec3{ 0, 0, 0 });
					break;
				}
			} else {
				// Spheres are ignored for now
				ray.origin = ray.origin + ray.direction * (nextHit.hitT + 0.0001f);
			}
		}
	}
}

void WireframeRenderer::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

void WireframeRenderer::load_scene(scene::SceneHandle scene, const ei::IVec2& resolution) {
	if(scene != m_currentScene) {
		m_currentScene = scene;
		m_sceneDesc = m_currentScene->get_descriptor<Device::CPU>({}, {}, {}, resolution);
		m_reset = true;
	}
}

} // namespace mufflon::renderer::silhouette