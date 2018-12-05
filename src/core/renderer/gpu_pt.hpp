#pragma once

#include "renderer.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/lights/light_tree.hpp"

namespace mufflon {

// Forward declarations
enum class Device : unsigned char;

namespace scene { namespace lights {
template < Device >
struct LightTree;
}}} // namespace mufflon::scene::lights

namespace mufflon { namespace renderer {

template < Device >
struct RenderBuffer;

class GpuPathTracer : public IRenderer {
public:
	GpuPathTracer(scene::SceneHandle scene);

	// This is just a test method, don't use this as an actual interface
	virtual void iterate(OutputHandler& handler) override;
	virtual void reset() override;
	virtual IParameterHandler& get_parameters() final { return m_params; } 

private:
	// Used so that we don't need to include everything in CU files
	void iterate(Pixel imageDims,
				 RenderBuffer<Device::CUDA> outputBuffer,
				 scene::lights::LightTree<Device::CUDA> lightTree) const;

	bool m_reset = true;
	ParameterHandler<PMinPathLength, PMaxPathLength, PNeeCount, PNeePositionGuide> m_params;
	scene::SceneHandle m_currentScene;
};

}} // namespace mufflon::renderer