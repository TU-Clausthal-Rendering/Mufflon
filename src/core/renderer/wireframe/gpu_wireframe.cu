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

using PtPathVertex = PathVertex<u8, 4>;

__global__ static void sample(RenderBuffer<Device::CUDA> outputBuffer,
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
	u8 vertexBuffer[256]; // TODO: depends on materials::MAX_MATERIAL_PARAMETER_SIZE
	PtPathVertex* vertex = as<PtPathVertex>(vertexBuffer);
	math::Rng rng(seeds[pixel]);
	// Create a start for the path
	int s = PtPathVertex::create_camera(vertex, vertex, scene->camera.get(), coord, rng.next());
	mAssertMsg(s < 256, "vertexBuffer overflow.");
	VertexSample sample = vertex->sample(scene->media, math::RndSet2_1{ rng.next(), rng.next() }, false);
	ei::Ray ray{ sample.origin, sample.excident };

#ifdef __CUDA_ARCH__
	while(true) {
		scene::accel_struct::RayIntersectionResult nextHit =
			scene::accel_struct::first_intersection_scene_lbvh<CURRENT_DEV>(*scene, ray, vertex->get_primitive_id(), scene::MAX_SCENE_SIZE);
		if(nextHit.hitId.instanceId < 0) {
			auto background = evaluate_background(scene->lightTree.background, ray.direction);
			if(any(greater(background.value, 0.0f))) {
				outputBuffer.contribute(coord, throughput, background.value,
										ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
										ei::Vec3{ 0, 0, 0 });
			}
			break;
		} else {
			const scene::LodDescriptor<CURRENT_DEV>& object = scene->lods[scene->lodIndices[nextHit.hitId.instanceId]];

			if(static_cast<u32>(nextHit.hitId.primId) < object.polygon.numTriangles) {
				float minBary = nextHit.surfaceParams.barycentric.x;
				minBary = minBary > nextHit.surfaceParams.barycentric.y ?
					nextHit.surfaceParams.barycentric.y : minBary;
				const float baryZ = 1.f - nextHit.surfaceParams.barycentric.x - nextHit.surfaceParams.barycentric.y;
				minBary = minBary > baryZ ? baryZ : minBary;

				float thickness = params.thickness;
				if(params.normalize) {
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

				if(minBary > params.thickness) {
					ray.origin = ray.origin + ray.direction * (nextHit.hitT + 0.0001f);
				} else {
					outputBuffer.contribute(coord, throughput, borderColor,
											ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
											ei::Vec3{ 0, 0, 0 });
					break;
				}
			} else if(static_cast<u32>(nextHit.hitId.primId) < object.polygon.numTriangles + object.polygon.numQuads) {
				float thickness = params.thickness;
				if(params.normalize) {
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
					outputBuffer.contribute(coord, throughput, borderColor,
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
#endif // __CUDA_ARCH__
}

namespace gpuwireframe_detail {

cudaError_t call_kernel(const dim3& gridDims, const dim3& blockDims,
						RenderBuffer<Device::CUDA>&& outputBuffer,
						scene::SceneDescriptor<Device::CUDA>* scene,
						const u32* seeds, const WireframeParameters& params) {
	sample<<<gridDims, blockDims>>>(std::move(outputBuffer), scene, seeds, params);
	return cudaGetLastError();
}

} // namespace gpuwireframe_detail

}
} // namespace mufflon::renderer
