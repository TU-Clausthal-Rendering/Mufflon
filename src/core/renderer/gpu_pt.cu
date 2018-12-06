#include "gpu_pt.hpp"
#include "output_handler.hpp"
#include "core/cuda/error.hpp"
#include "core/math/rng.hpp"
#include "core/scene/lights/light_tree.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <random>
#include "core/scene/textures/interface.hpp"

using namespace mufflon::scene::lights;

namespace mufflon { namespace renderer {

__global__ static void sample(Pixel imageDims,
							  RenderBuffer<Device::CUDA> outputBuffer,
							  LightTree<Device::CUDA>* lightTree,
							  const float* rnds) {
	Pixel coord{
		threadIdx.x + blockDim.x * blockIdx.x,
		threadIdx.y + blockDim.y * blockIdx.y
	};

	//RndSet rnd{ 1, 2, 3, 4 };
	//Photon p = emit(lightTree, 1u, 2u, 12345u, ei::Box{ ei::Vec3{0,0,0},ei::Vec3{1,1,1} }, rnd);
	//printf("PDF: %f\n", static_cast<float>(p.dir.pdf));

	Throughput throughput{ ei::Vec3{1.0f}, 1.0f };
	int pathLen = 0;
	do {
		if(pathLen < 666) { // TODO: min and max pathLen bounds

			// TODO: Call NEE member function for the camera start/recursive vertices
			// TODO: Walk
			//head = walk(head);
			break;
		}
		++pathLen;
	} while(pathLen < 666);

	// Random walk ended because of missing the scene?
	if(pathLen < 666) {
		if(coord.x < imageDims.x && coord.y < imageDims.y) {
			const float phi = 2.f * ei::PI * (coord.x + 0.5f) / static_cast<float>(imageDims.x);
			const float theta = ei::PI * (coord.y + 0.5f) / static_cast<float>(imageDims.y);
			ei::Vec3 testRadiance = lightTree->background.get_color(ei::Vec3{
				sinf(theta) * cosf(phi),
				sinf(theta) * sinf(phi),
				cosf(theta)
			});
			outputBuffer.contribute(coord, throughput, testRadiance,
									ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
									ei::Vec3{ 0, 0, 0 });
		}
	}
}

void GpuPathTracer::iterate(Pixel imageDims,
							RenderBuffer<Device::CUDA> outputBuffer,
							LightTree<Device::CUDA> lightTree) const {
	
	std::unique_ptr<float[]> rnds = std::make_unique<float[]>(2 * imageDims.x * imageDims.y);
	math::Xoroshiro128 rng{ static_cast<u32>(std::random_device()()) };
	for (int i = 0; i < 2*imageDims.x*imageDims.y; ++i) {
		rnds[i] = static_cast<u32>(rng.next()) / static_cast<float>(std::numeric_limits<u32>::max());
	}
	float* devRnds = nullptr;
	cuda::check_error(cudaMalloc(&devRnds, sizeof(float) * 2 * imageDims.x * imageDims.y));
	cuda::check_error(cudaMemcpy(devRnds, rnds.get(), sizeof(float) * 2 * imageDims.x * imageDims.y,
		cudaMemcpyHostToDevice));

	// TODO: pass scene data to kernel!
	dim3 blockDims{ 16u, 16u, 1u };
	dim3 gridDims{
		1u + static_cast<u32>(imageDims.x - 1) / blockDims.x,
		1u + static_cast<u32>(imageDims.y - 1) / blockDims.y,
		1u
	};

	// TODO
	LightTree<Device::CUDA>* cudaLightTree = nullptr;
	cuda::check_error(cudaMalloc(&cudaLightTree, sizeof(lightTree)));
	cuda::check_error(cudaMemcpy(cudaLightTree, &lightTree, sizeof(lightTree), cudaMemcpyHostToDevice));
	cuda::check_error(cudaGetLastError());
	sample<<<gridDims, blockDims>>>(imageDims,
									std::move(outputBuffer),
									cudaLightTree, devRnds);
	cuda::check_error(cudaGetLastError());
	cuda::check_error(cudaFree(cudaLightTree));
	cuda::check_error(cudaFree(devRnds));
}

}} // namespace mufflon::renderer