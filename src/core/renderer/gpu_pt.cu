#include "gpu_pt.hpp"
#include "output_handler.hpp"
#include "core/cuda/error.hpp"
#include "core/math/rng.hpp"
#include "core/scene/lights/light_tree.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <random>

using namespace mufflon::scene::lights;

namespace mufflon { namespace renderer {

__global__ static void sample(Pixel imageDims,
							  LightTree<Device::CUDA> lightTree,
							  RenderBuffer<Device::CUDA> outputBuffer,
							  const float* rnds) {
	Pixel coord{
		threadIdx.x + blockDim.x * blockIdx.x,
		threadIdx.y + blockDim.y * blockIdx.y
	};

	//RndSet rnd{ 1, 2, 3, 4 };
	//Photon p = emit(lightTree, 1u, 2u, 12345u, ei::Box{ ei::Vec3{0,0,0},ei::Vec3{1,1,1} }, rnd);
	//printf("PDF: %f\n", static_cast<float>(p.dir.pdf));

	Throughput throughput {ei::Vec3{1.0f}, 1.0f};
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
		constexpr ei::Vec3 colors[4]{
				ei::Vec3{0, 0, 1},
				ei::Vec3{0, 1, 0},
				ei::Vec3{1, 0, 0},
				ei::Vec3{1, 1, 1}
		};
		if (coord.x < imageDims.x && coord.y < imageDims.y) {
			float x = coord.x / static_cast<float>(imageDims.x);
			float y = coord.y / static_cast<float>(imageDims.y);
			x *= rnds[2 * (coord.x + coord.y * imageDims.x)];
			y *= rnds[2 * (coord.x + coord.y * imageDims.x) + 1];
			ei::Vec3 testRadiance = colors[0u] * (1.f - x)*(1.f - y) + colors[1u] * x*(1.f - y)
				+ colors[2u] * (1.f - x)*y + colors[3u] * x*y;
			outputBuffer.contribute(coord, throughput, testRadiance,
									ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
									ei::Vec3{ 0, 0, 0 });
		}
	}
}

void GpuPathTracer::iterate(Pixel imageDims,
							scene::lights::LightTree<Device::CUDA> lightTree,
							RenderBuffer<Device::CUDA> outputBuffer) const {
	// TODO: remove, only for debugging
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

	cuda::check_error(cudaPeekAtLastError());
	sample<<<gridDims, blockDims>>>(imageDims, std::move(lightTree),
										 std::move(outputBuffer), devRnds);
	cuda::check_error(cudaGetLastError());
	cuda::check_error(cudaFree(devRnds));
}

}} // namespace mufflon::renderer