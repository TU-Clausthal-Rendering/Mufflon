#pragma once

#include "export/api.hpp"
#include "core/scene/lights/light_tree.hpp"

extern "C" LIBRARY_API void test_gpu_pt();

namespace mufflon { namespace renderer {

class GpuPathTracer {
public:
	// This is just a test method, don't use this as an actual interface
	__host__ void run();

private:
	mufflon::scene::lights::LightTree<mufflon::Device::CUDA> m_lights;
};

}} // namespace mufflon::renderer