#include "gpu_pt.hpp"
#include <cuda_runtime.h>

using namespace mufflon::scene::lights;

namespace mufflon { namespace renderer {

__global__ static void runPt(LightTree<mufflon::Device::CUDA> tree) {
	RndSet rnd{ 1, 2, 3, 4 };
	Photon p = emit(tree, 1u, 2u, 12345u, ei::Box{ ei::Vec3{0,0,0},ei::Vec3{1,1,1} }, rnd);
	printf("PDF: %f\n", static_cast<float>(p.dir.pdf));
}

void GpuPathTracer::run() {
	runPt<<<1, 1>>>(this->m_lights);
}



}} // namespace mufflon::renderer