#pragma once

#include "export/api.hpp"
#include "core/scene/lights/light_tree.hpp"

namespace mufflon { namespace renderer {

class LIBRARY_API GpuPathTracer {
public:
	// This is just a test method, don't use this as an actual interface
	__host__ void run();

private:
	mufflon::scene::lights::LightTree<mufflon::Device::CUDA> m_lights;
};

}} // namespace mufflon::renderer