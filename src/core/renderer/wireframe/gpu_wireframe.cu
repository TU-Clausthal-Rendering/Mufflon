#include "wireframe_params.hpp"
#include "core/math/rng.hpp"
#include "core/memory/residency.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/path_util.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_sampling.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <ei/vector.hpp>
#include <random>

using namespace mufflon::scene::lights;

namespace mufflon {
namespace renderer {

using PtPathVertex = PathVertex<VertexExtension>;

namespace {

CUDA_FUNCTION ei::Vec3 translateToWorldSpace(const scene::SceneDescriptor<Device::CUDA>& scene, const ei::Vec3& point,
											 const i32 instanceId) {
	const ei::Mat3x3 rotation{ scene.transformations[instanceId] };
	const ei::Vec3 scale{ scene.scales[instanceId] };
	const ei::Vec3 translation{
		scene.transformations[instanceId][3],
		scene.transformations[instanceId][7],
		scene.transformations[instanceId][11]
	};
	return rotation * (point * scale) + translation;
}

CUDA_FUNCTION ei::Vec3 computeClosestLinePoint(const scene::SceneDescriptor<Device::CUDA>& scene, const i32 instanceId,
											   const ei::IVec3& indices, const ei::Vec3& hitpoint) {
	// Compute the projected points on the triangle lines
	const auto& vertices = scene.lods[scene.lodIndices[instanceId]].polygon.vertices;
	const auto A = translateToWorldSpace(scene, vertices[indices.x], instanceId);
	const auto B = translateToWorldSpace(scene, vertices[indices.y], instanceId);
	const auto C = translateToWorldSpace(scene, vertices[indices.z], instanceId);
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

CUDA_FUNCTION ei::Vec3 computeClosestLinePoint(const scene::SceneDescriptor<Device::CUDA>& scene, const i32 instanceId,
											   const ei::IVec4& indices, const ei::Vec3& hitpoint) {
	// Compute the projected points on the quad lines
	const auto& vertices = scene.lods[scene.lodIndices[instanceId]].polygon.vertices;
	const auto A = translateToWorldSpace(scene, vertices[indices.x], instanceId);
	const auto B = translateToWorldSpace(scene, vertices[indices.y], instanceId);
	const auto C = translateToWorldSpace(scene, vertices[indices.z], instanceId);
	const auto D = translateToWorldSpace(scene, vertices[indices.w], instanceId);
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

CUDA_FUNCTION ei::Vec3 computeClosestLinePoint(const scene::SceneDescriptor<Device::CUDA>& scene, const i32 instanceId,
											   const u32 index, const ei::Vec3& hitpoint, const ei::Vec3& incident) {
	const auto& sphere = scene.lods[scene.lodIndices[instanceId]].spheres.spheres[index];
	const auto center = translateToWorldSpace(scene, sphere.center, instanceId);
	// First we compute the vector pointing to the edge of the sphere from our point-of-view
	const auto centerToHit = hitpoint - center;
	const auto down = ei::cross(centerToHit, incident);
	const auto centerToRim = ei::normalize(ei::cross(incident, down));
	// Now scale with radius -> done (TODO: scale radius? probably, but don't wanna check right now)
	return centerToRim * sphere.radius;
}

} // namespace 

__global__ static void sample_wireframe(RenderBuffer<Device::CUDA> outputBuffer,
										scene::SceneDescriptor<Device::CUDA>* scene,
										const u32* seeds, WireframeParameters params) {
	Pixel coord{
		threadIdx.x + blockDim.x * blockIdx.x,
		threadIdx.y + blockDim.y * blockIdx.y
	};
	if(coord.x >= outputBuffer.get_width() || coord.y >= outputBuffer.get_height())
		return;

	const int pixel = coord.x + coord.y * outputBuffer.get_width();

	constexpr ei::Vec3 borderColor{ 1.f };

	math::Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	PtPathVertex vertex;
	math::Rng rng(seeds[pixel]);
	// Create a start for the path
	PtPathVertex::create_camera(&vertex, &vertex, scene->camera.get(), coord, rng.next());
	VertexSample sample = vertex.sample(scene->media, math::RndSet2_1{ rng.next(), rng.next() }, false);
	ei::Ray ray{ sample.origin, sample.excident };

#ifdef __CUDA_ARCH__
	while(true) {
		scene::accel_struct::RayIntersectionResult nextHit =
			scene::accel_struct::first_intersection(*scene, ray, vertex.get_primitive_id(), scene::MAX_SCENE_SIZE);
		if(nextHit.hitId.instanceId < 0) {
			auto background = evaluate_background(scene->lightTree.background, ray.direction);
			if(any(greater(background.value, 0.0f))) {
				outputBuffer.contribute(coord, throughput, background.value,
										ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
										ei::Vec3{ 0, 0, 0 });
			}
			break;
		} else {
			const auto& hitId = nextHit.hitId;
			const auto& lod = scene->lods[scene->lodIndices[hitId.instanceId]];
			const auto& poly = lod.polygon;
			const auto& hit = ray.origin + ray.direction * nextHit.hitT;

			ei::Vec3 closestLinePoint;
			if(static_cast<u32>(nextHit.hitId.primId) < poly.numTriangles) {
				const ei::IVec3 indices{
					poly.vertexIndices[3 * hitId.primId + 0],
					poly.vertexIndices[3 * hitId.primId + 1],
					poly.vertexIndices[3 * hitId.primId + 2]
				};
				closestLinePoint = computeClosestLinePoint(*scene, hitId.instanceId, indices, hit);
			} else if(static_cast<u32>(nextHit.hitId.primId < (poly.numTriangles + poly.numQuads))) {
				const ei::IVec4 indices{
					poly.vertexIndices[3 * poly.numTriangles + 4 * hitId.primId + 0],
					poly.vertexIndices[3 * poly.numTriangles + 4 * hitId.primId + 1],
					poly.vertexIndices[3 * poly.numTriangles + 4 * hitId.primId + 2],
					poly.vertexIndices[3 * poly.numTriangles + 4 * hitId.primId + 3]
				};
				closestLinePoint = computeClosestLinePoint(*scene, hitId.instanceId, indices, hit);
			} else {
				closestLinePoint = computeClosestLinePoint(*scene, hitId.instanceId, hitId.primId, hit, ray.direction);
			}

			// If the point is within x pixels of the line we paint
			const auto& camParams = scene->camera.get();
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
			if(ei::max(ei::abs(projectedPixel.x - coord.x), ei::abs(projectedPixel.y - coord.y)) > params.lineWidth) {
				ray.origin = ray.origin + ray.direction * (nextHit.hitT + 0.0001f);
			} else {
				outputBuffer.contribute(coord, throughput, borderColor,
										ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
										ei::Vec3{ 0, 0, 0 });
				break;
			}
		}
	}
#endif // __CUDA_ARCH__
}

namespace gpuwireframe_detail {

cudaError_t call_kernel(const dim3& gridDims, const dim3& blockDims,
						RenderBuffer<Device::CUDA>&& outputBuffer,
						scene::SceneDescriptor<Device::CUDA>* scene,
						const u32* seeds, const WireframeParameters& params) {
	sample_wireframe <<<gridDims, blockDims>>>(std::move(outputBuffer), scene, seeds, params);
	return cudaGetLastError();
}

} // namespace gpuwireframe_detail

}
} // namespace mufflon::renderer
