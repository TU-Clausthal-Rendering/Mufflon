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

void CpuWireframe::on_descriptor_requery() {
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
			scene::accel_struct::first_intersection(m_sceneDesc, ray, vertex.get_primitive_id(), scene::MAX_SCENE_SIZE);
		if(nextHit.hitId.instanceId < 0) {
			auto background = evaluate_background(m_sceneDesc.lightTree.background, ray.direction);
			if(any(greater(background.value, 0.0f))) {
				m_outputBuffer.contribute(coord, throughput, background.value,
										ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
										ei::Vec3{ 0, 0, 0 });
			}
			break;
		} else {
			const auto& hitId = nextHit.hitId;
			const auto& lod = m_sceneDesc.lods[m_sceneDesc.lodIndices[hitId.instanceId]];
			const auto& poly = lod.polygon;
			const auto& hit = ray.origin + ray.direction * nextHit.hitT;

			ei::Vec3 closestLinePoint;
			if(static_cast<u32>(nextHit.hitId.primId) < poly.numTriangles) {
				const ei::IVec3 indices{
					poly.vertexIndices[3 * hitId.primId + 0],
					poly.vertexIndices[3 * hitId.primId + 1],
					poly.vertexIndices[3 * hitId.primId + 2]
				};
				closestLinePoint = computeClosestLinePoint(hitId.instanceId, indices, hit);
			} else if(static_cast<u32>(nextHit.hitId.primId < (poly.numTriangles + poly.numQuads))) {
				const ei::IVec4 indices{
					poly.vertexIndices[3 * poly.numTriangles + 4 * hitId.primId + 0],
					poly.vertexIndices[3 * poly.numTriangles + 4 * hitId.primId + 1],
					poly.vertexIndices[3 * poly.numTriangles + 4 * hitId.primId + 2],
					poly.vertexIndices[3 * poly.numTriangles + 4 * hitId.primId + 3]
				};
				closestLinePoint = computeClosestLinePoint(hitId.instanceId, indices, hit);
			} else {
				closestLinePoint = computeClosestLinePoint(hitId.instanceId, hitId.primId, hit, ray.direction);
			}

			// If the point is within x pixels of the line we paint (TODO: how to perform anti-aliasing?)
			const auto& camParams = m_sceneDesc.camera.get();
			ei::IVec2 projectedPixel;
			switch(camParams.type) {
				case cameras::CameraModel::PINHOLE:
					projectedPixel = cameras::pinholecam_project(static_cast<const cameras::PinholeParams&>(camParams),
																 ei::normalize(closestLinePoint - ray.origin)).coord;
					break;
				case cameras::CameraModel::FOCUS:	// TODO: does this make sense?
				case cameras::CameraModel::ORTHOGRAPHIC:
				default:
					mAssertMsg(false, "Unknown or invalid camera model");
			}
			if(ei::max(ei::abs(projectedPixel.x - coord.x), ei::abs(projectedPixel.y - coord.y)) > m_params.lineWidth) {
				ray.origin = ray.origin + ray.direction * (nextHit.hitT + 0.0001f);
			} else {
				m_outputBuffer.contribute(coord, throughput, borderColor,
										  ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
										  ei::Vec3{ 0, 0, 0 });
				break;
			}
		}
	}
}

ei::Vec3 CpuWireframe::translateToWorldSpace(const ei::Vec3& point, const i32 instanceId) const {
	const ei::Mat3x3 rotation{ m_sceneDesc.transformations[instanceId] };
	const ei::Vec3 scale{ m_sceneDesc.scales[instanceId] };
	const ei::Vec3 translation{
		m_sceneDesc.transformations[instanceId][3],
		m_sceneDesc.transformations[instanceId][7],
		m_sceneDesc.transformations[instanceId][11]
	};
	return rotation * (point * scale) + translation;
}

ei::Vec3 CpuWireframe::computeClosestLinePoint(const i32 instanceId, const ei::IVec3& indices, const ei::Vec3& hitpoint) const {
	// Compute the projected points on the triangle lines
	const auto& vertices = m_sceneDesc.lods[m_sceneDesc.lodIndices[instanceId]].polygon.vertices;
	const auto A = translateToWorldSpace(vertices[indices.x], instanceId);
	const auto B = translateToWorldSpace(vertices[indices.y], instanceId);
	const auto C = translateToWorldSpace(vertices[indices.z], instanceId);
	const auto AB = B - A;
	const auto AC = C - A;
	const auto BC = C - B;
	const auto AP = hitpoint - A;
	const auto BP = hitpoint - B;
	const auto onAB = A + ei::dot(AP, AB) / ei::lensq(AB) * AB;
	const auto onAC = A + ei::dot(AP, AC) / ei::lensq(AC) * AC;
	const auto onBC = B + ei::dot(BP, BC) / ei::lensq(BC) * BC;

	// Determine the point closest to the hitpoint
	const auto distAB = ei::lensq(onAB - hitpoint);
	const auto distAC = ei::lensq(onAC - hitpoint);
	const auto distBC = ei::lensq(onBC - hitpoint);
	ei::Vec3 closestLinePoint;
	if(distAB <= distAC && distAB <= distBC)
		return onAB;
	else if(distAC <= distAB && distAC <= distBC)
		return onAC;
	else
		return onBC;
}

ei::Vec3 CpuWireframe::computeClosestLinePoint(const i32 instanceId, const ei::IVec4& indices, const ei::Vec3& hitpoint) const {
	// Compute the projected points on the quad lines
	const auto& vertices = m_sceneDesc.lods[m_sceneDesc.lodIndices[instanceId]].polygon.vertices;
	const auto A = translateToWorldSpace(vertices[indices.x], instanceId);
	const auto B = translateToWorldSpace(vertices[indices.y], instanceId);
	const auto C = translateToWorldSpace(vertices[indices.z], instanceId);
	const auto D = translateToWorldSpace(vertices[indices.w], instanceId);
	const auto AB = B - A;
	const auto AC = C - A;
	const auto BD = D - B;
	const auto CD = D - C;
	const auto AP = hitpoint - A;
	const auto BP = hitpoint - B;
	const auto CP = hitpoint - C;
	const auto onAB = A + ei::dot(AP, AB) / ei::lensq(AB) * AB;
	const auto onAC = A + ei::dot(AP, AC) / ei::lensq(AC) * AC;
	const auto onBD = B + ei::dot(BP, BD) / ei::lensq(BD) * BD;
	const auto onCD = C + ei::dot(CP, CD) / ei::lensq(CD) * CD;

	// Determine the point closest to the hitpoint
	const auto distAB = ei::lensq(onAB - hitpoint);
	const auto distAC = ei::lensq(onAC - hitpoint);
	const auto distBD = ei::lensq(onBD - hitpoint);
	const auto distCD = ei::lensq(onCD - hitpoint);
	ei::Vec3 closestLinePoint;
	if(distAB <= distAC && distAB <= distBD && distAB <= distCD)
		return onAB;
	else if(distAC <= distAB && distAC <= distBD && distAC <= distCD)
		return onAC;
	else if(distBD <= distAB && distBD <= distAC && distBD <= distCD)
		return onBD;
	else
		return onCD;
}

ei::Vec3 CpuWireframe::computeClosestLinePoint(const i32 instanceId, const u32 index, const ei::Vec3& hitpoint,
											   const ei::Vec3& incident) const {
	const auto& sphere = m_sceneDesc.lods[m_sceneDesc.lodIndices[instanceId]].spheres.spheres[index];
	const auto center = translateToWorldSpace(sphere.center, instanceId);
	// First we compute the vector pointing to the edge of the sphere from our point-of-view
	const auto centerToHit = hitpoint - center;
	const auto down = ei::cross(centerToHit, incident);
	const auto centerToRim = ei::normalize(ei::cross(incident, down));
	// Now scale with radius -> done (TODO: scale radius? probably, but don't wanna check right now)
	return centerToRim * sphere.radius;
}

void CpuWireframe::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer