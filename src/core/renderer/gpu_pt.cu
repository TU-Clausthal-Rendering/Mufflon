#include "gpu_pt.hpp"
#include <cuda_runtime.h>

#include "core/scene/textures/interface.hpp"

using namespace mufflon::scene::lights;

namespace mufflon { namespace renderer {

__global__ static void runPt(LightTree<mufflon::Device::CUDA> tree) {
	RndSet rnd{ 1, 2, 3, 4 };
	Photon p = emit(tree, 1u, 2u, 12345u, ei::Box{ ei::Vec3{0,0,0},ei::Vec3{1,1,1} }, rnd);
	printf("PDF: %f\n", static_cast<float>(p.dir.pdf));

	/*read(tex, Pixel{0,0});
	sample(tex, ei::Vec2{0.0f});
	read(surf, Pixel{0,0});
	write(surf, ei::Vec4{1.0f}, Pixel{0,0});*/
}

void GpuPathTracer::run() {
	/*scene::textures::Texture* tex;
	auto texCPU = *tex->aquireConst<Device::CPU>();
	read(texCPU, Pixel{0,0});
	sample(texCPU, ei::Vec2{0.5f});
	runPt<<<1, 1>>>(this->m_lights, *tex->aquireConst<Device::CUDA>(), *tex->aquire<Device::CUDA>());*/
	runPt<<<1, 1>>>(this->m_lights);
}



}} // namespace mufflon::renderer