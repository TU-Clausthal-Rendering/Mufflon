#include "cpu_wireframe.hpp"
#include "util/parallel.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/scene/scene.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include <random>

namespace mufflon::renderer {

using PtPathVertex = PathVertex<VertexExtension>;

CpuWireframe::CpuWireframe() {
	// TODO: init one RNG per thread?
	m_rngs.emplace_back(static_cast<u32>(std::random_device()()));

	// The PT does not need additional memory resources like photon maps.
}

void CpuWireframe::post_descriptor_requery() {
	init_rngs(m_outputBuffer.get_num_pixels());
}

void CpuWireframe::iterate() {
	// TODO: call sample in a parallel way for each output pixel
	// TODO: better pixel order?
	// TODO: different scheduling?
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
		this->sample(Pixel{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() });
	}
}

void CpuWireframe::sample(const Pixel coord) {
	int pixel = coord.x + coord.y * m_outputBuffer.get_width();

	constexpr ei::Vec3 borderColor{ 1.f };

	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	PtPathVertex vertex;
	// Create a start for the path
	PtPathVertex::create_camera(&vertex, &vertex, m_sceneDesc.camera.get(), coord, m_rngs[pixel].next());

	math::RndSet2_1 rnd{ m_rngs[pixel].next(), m_rngs[pixel].next() };
	VertexSample sample = vertex.sample(m_sceneDesc.media, rnd, false);
	ei::Ray ray{ sample.origin, sample.excident };

	while(true) {
		scene::accel_struct::RayIntersectionResult nextHit =
			scene::accel_struct::first_intersection_scene_lbvh<CURRENT_DEV>(m_sceneDesc, ray, vertex.get_primitive_id(), scene::MAX_SCENE_SIZE);
		if(nextHit.hitId.instanceId < 0) {
			auto background = evaluate_background(m_sceneDesc.lightTree.background, ray.direction);
			if(any(greater(background.value, 0.0f))) {
				m_outputBuffer.contribute(coord, throughput, background.value,
										ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
										ei::Vec3{ 0, 0, 0 });
			}
			break;
		} else {
			const scene::LodDescriptor<CURRENT_DEV>& object = m_sceneDesc.lods[m_sceneDesc.lodIndices[nextHit.hitId.instanceId]];

			if(static_cast<u32>(nextHit.hitId.primId) < object.polygon.numTriangles) {
				float minBary = nextHit.surfaceParams.barycentric.x;
				minBary = minBary > nextHit.surfaceParams.barycentric.y ?
					nextHit.surfaceParams.barycentric.y : minBary;
				const float baryZ = 1.f - nextHit.surfaceParams.barycentric.x - nextHit.surfaceParams.barycentric.y;
				minBary = minBary > baryZ ? baryZ : minBary;

				float thickness = m_params.thickness;
				if(m_params.normalize) {
					const ei::IVec3 idx{
						object.polygon.vertexIndices[3 * nextHit.hitId.primId + 0],
						object.polygon.vertexIndices[3 * nextHit.hitId.primId + 1],
						object.polygon.vertexIndices[3 * nextHit.hitId.primId + 2]
					};
					const float area = ei::surface(ei::Triangle{
						object.polygon.vertices[idx.x],
						object.polygon.vertices[idx.y],
						object.polygon.vertices[idx.z]
					});
					thickness /= area;
				}

				if(minBary > m_params.thickness) {
					ray.origin = ray.origin + ray.direction * (nextHit.hitT + 0.0001f);
				} else {
					m_outputBuffer.contribute(coord, throughput, borderColor,
											ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
											ei::Vec3{ 0, 0, 0 });
					break;
				}
			} else if(static_cast<u32>(nextHit.hitId.primId) < object.polygon.numTriangles + object.polygon.numQuads) {
				float thickness = m_params.thickness;
				if(m_params.normalize) {
					const i32 quadId = nextHit.hitId.primId - object.polygon.numTriangles;
					const ei::IVec4 idx{
						object.polygon.vertexIndices[3 * object.polygon.numTriangles + 4 * quadId + 0],
						object.polygon.vertexIndices[3 * object.polygon.numTriangles + 4 * quadId + 1],
						object.polygon.vertexIndices[3 * object.polygon.numTriangles + 4 * quadId + 2],
						object.polygon.vertexIndices[3 * object.polygon.numTriangles + 4 * quadId + 3]
					};
					const float area = ei::surface(ei::Triangle{
						object.polygon.vertices[idx.x],
						object.polygon.vertices[idx.y],
						object.polygon.vertices[idx.z]
					 }) + ei::surface(ei::Triangle{
						object.polygon.vertices[idx.x],
						object.polygon.vertices[idx.z],
						object.polygon.vertices[idx.w]
					});
					thickness /= area;
				}

				if((nextHit.surfaceParams.bilinear.x > thickness && nextHit.surfaceParams.bilinear.x < 1.f - thickness)
				   && (nextHit.surfaceParams.bilinear.y > thickness && nextHit.surfaceParams.bilinear.y < 1.f - thickness)) {
					ray.origin = ray.origin + ray.direction * (nextHit.hitT + 0.0001f);
				} else {
					m_outputBuffer.contribute(coord, throughput, borderColor,
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

void CpuWireframe::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer