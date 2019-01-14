#include "gpu_pt.hpp"
#include "output_handler.hpp"
#include "core/cuda/error.hpp"
#include "core/math/rng.hpp"
#include "core/scene/lights/light_tree.hpp"
#include "core/scene/textures/interface.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include "path_util.hpp"
#include "random_walk.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <random>

using namespace mufflon::scene::lights;

namespace mufflon { namespace renderer {

using PtPathVertex = PathVertex<u8, 4>;

__global__ static void sample(RenderBuffer<Device::CUDA> outputBuffer,
							  scene::SceneDescriptor<Device::CUDA>* scene,
							  const u32* seeds) {
	Pixel coord{
		threadIdx.x + blockDim.x * blockIdx.x,
		threadIdx.y + blockDim.y * blockIdx.y
	};
	if(coord.x >= outputBuffer.get_width() || coord.y >= outputBuffer.get_height())
		return;
	const u32 maxPathLength = 4u;

	int pixel = coord.x + coord.y * outputBuffer.get_width();

	math::Rng rng(seeds[pixel]);

	Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	u8 vertexBuffer[256]; // TODO: depends on materials::MAX_MATERIAL_PARAMETER_SIZE
	PtPathVertex* vertex = as<PtPathVertex>(vertexBuffer);
	// Create a start for the path
	int s = PtPathVertex::create_camera(vertex, vertex, scene->camera.get(), coord, rng.next());
	mAssertMsg(s < 256, "vertexBuffer overflow.");


#ifdef __CUDA_ARCH__
	int pathLen = 0;
	
	do {
		if(pathLen > 0 && pathLen + 1 <= maxPathLength) {
			// Call NEE member function for recursive vertices.
			// Do not connect to the camera, because this makes the renderer much more
			// complicated. Our decision: The PT should be as simple as possible!
			// What means more complicated?
			// A connnection to the camera results in a different pixel. In a multithreaded
			// environment this means that we need a write mutex for each pixel.
			// TODO: test/parametrize mulievent estimation (more indices in connect) and different guides.
			u64 neeSeed = rng.next();
			math::RndSet2 neeRnd = rng.next();
			auto nee = connect(scene->lightTree, 0, 1, neeSeed,
							   vertex->get_position(), scene->aabb,
							   neeRnd, scene::lights::guide_flux);
			auto value = vertex->evaluate(nee.direction, scene->media);
			mAssert(!isnan(value.value.x) && !isnan(value.value.y) && !isnan(value.value.z));
			Spectrum radiance = value.value * nee.diffIrradiance;
			if(any(greater(radiance, 0.0f)) && value.cosOut > 0.0f) {
				bool anyhit = scene::accel_struct::any_intersection_scene_lbvh<Device::CUDA>(
					*scene, { vertex->get_position() , nee.direction },
					vertex->get_primitive_id(), nee.dist);
				if(!anyhit) {
					AreaPdf hitPdf = value.pdfF.to_area_pdf(nee.cosOut, nee.distSq);
					float mis = 1.0f / (1.0f + hitPdf / nee.creationPdf);
					mAssert(!isnan(mis));
					outputBuffer.contribute(coord, throughput, { Spectrum{1.0f}, 1.0f },
											value.cosOut, radiance * mis);
				}
			}
		}

		// Walk
		scene::Point lastPosition = vertex->get_position();
		math::RndSet2_1 rnd{ rng.next(), rng.next() };
		math::DirectionSample lastDir;
		if(!walk(*scene, *vertex, rnd, -1.0f, false, throughput, vertex, lastDir)) {
			if(throughput.weight != Spectrum{ 0.f }) {
				// Missed scene - sample background
				auto background = evaluate_background(scene->lightTree.background, lastDir.direction);
				if(any(greater(background.value, 0.0f))) {
					float mis = 1.0f / (1.0f + background.pdfB / lastDir.pdf);
					background.value *= mis;
					outputBuffer.contribute(coord, throughput, background.value,
											ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
											ei::Vec3{ 0, 0, 0 });
				}
			}
			break;
		}
		++pathLen;

		// Evaluate direct hit of area ligths
		if(pathLen <= maxPathLength) {
			Spectrum emission = vertex->get_emission();
			if(emission != 0.0f) {
				AreaPdf backwardPdf = connect_pdf(scene->lightTree, 0,
												  lastPosition, scene::lights::guide_flux);
				//float mis = 1.0f / (1.0f + backwardPdf / vertex->get_incident_pdf());
				float mis = 0.0f;
				outputBuffer.contribute(coord, throughput, emission, vertex->get_position(),
										vertex->get_normal(), vertex->get_albedo());
			}
		}
	} while(pathLen < maxPathLength);
#endif // __CUDA_ARCH__
}

void GpuPathTracer::iterate(Pixel imageDims,
							RenderBuffer<Device::CUDA> outputBuffer) const {
	
	std::unique_ptr<u32[]> rnds = std::make_unique<u32[]>(imageDims.x * imageDims.y);
	math::Xoroshiro128 rng{ static_cast<u32>(std::random_device()()) };
	for (int i = 0; i < imageDims.x*imageDims.y; ++i)
		rnds[i] = static_cast<u32>(rng.next());
	u32* devRnds = nullptr;
	cuda::check_error(cudaMalloc(&devRnds, sizeof(u32) * imageDims.x * imageDims.y));
	cuda::check_error(cudaMemcpy(devRnds, rnds.get(), sizeof(u32) * imageDims.x * imageDims.y,
		cudaMemcpyDefault));

	// TODO: pass scene data to kernel!
	dim3 blockDims{ 16u, 16u, 1u };
	dim3 gridDims{
		1u + static_cast<u32>(imageDims.x - 1) / blockDims.x,
		1u + static_cast<u32>(imageDims.y - 1) / blockDims.y,
		1u
	};

	// TODO
	cuda::check_error(cudaGetLastError());
	sample<<<gridDims, blockDims>>>(std::move(outputBuffer),
									m_scenePtr, devRnds);
	cuda::check_error(cudaGetLastError());
	cuda::check_error(cudaFree(devRnds));
}

}} // namespace mufflon::renderer