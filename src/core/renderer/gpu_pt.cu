#include "gpu_pt.hpp"
#include "core/scene/lights/light_tree.hpp"
#include "output_handler.hpp"
#include <cuda_runtime.h>

using namespace mufflon::scene::lights;

namespace mufflon { namespace renderer {

__global__ static void sample(Pixel imageDims,
							  LightTree<Device::CUDA> lightTree,
							  RenderBuffer<Device::CUDA> outputBuffer) {
	Pixel coord{
		threadIdx.x + blockDim.x * blockIdx.x,
		threadIdx.y + blockDim.y * blockIdx.y
	};

	//RndSet rnd{ 1, 2, 3, 4 };
	//Photon p = emit(lightTree, 1u, 2u, 12345u, ei::Box{ ei::Vec3{0,0,0},ei::Vec3{1,1,1} }, rnd);
	//printf("PDF: %f\n", static_cast<float>(p.dir.pdf));

	PathHead head{
		Throughput{ei::Vec3{1.0f}, 1.0f}
	};

	int pathLen = 0;
	do {
		if(pathLen < 666) { // TODO: min and max pathLen bounds

			// TODO: Call NEE member function for the camera start/recursive vertices
			// TODO: Walk
			//head = walk(head);
			break;
		}
		++pathLen;
	} while(pathLen < 666 && head.type != Interaction::VOID);

	// Random walk ended because of missing the scene?
	if(pathLen < 666 && head.type == Interaction::VOID) {
		constexpr ei::Vec3 colors[4]{
				ei::Vec3{1, 0, 0},
				ei::Vec3{0, 1, 0},
				ei::Vec3{0, 0, 1},
				ei::Vec3{1, 1, 1}
		};
		float x = coord.x / static_cast<float>(imageDims.x);
		float y = coord.y / static_cast<float>(imageDims.y);
		ei::Vec3 testRadiance = colors[0u] * (1.f - x)*(1.f - y) + colors[1u] * x*(1.f - y)
			+ colors[2u] * (1.f - x)*y + colors[3u] * x*y;
		if(coord.x < imageDims.x && coord.y < imageDims.y) {
			outputBuffer.contribute(coord, head.throughput, testRadiance,
									ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
									ei::Vec3{ 0, 0, 0 });
		}
	}
}

void GpuPathTracer::iterate(Pixel imageDims,
							scene::lights::LightTree<Device::CUDA> lightTree,
							RenderBuffer<Device::CUDA> outputBuffer) const {
	// TODO: pass scene data to kernel!
	sample<<<imageDims.x, imageDims.y>>>(imageDims, std::move(lightTree),
											  std::move(outputBuffer));
}

}} // namespace mufflon::renderer