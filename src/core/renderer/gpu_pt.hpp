#pragma once

#include "renderer.hpp"
#include "core/scene/descriptors.hpp"
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
	GpuPathTracer();
	~GpuPathTracer();

	// This is just a test method, don't use this as an actual interface
	virtual void iterate(OutputHandler& handler) override;
	virtual void reset() override;
	virtual IParameterHandler& get_parameters() final { return m_params; }
	virtual bool has_scene() const noexcept override { return m_currentScene != nullptr; }
	virtual void load_scene(scene::SceneHandle scene, const ei::IVec2& resolution) override;

private:
	// Used so that we don't need to include everything in CU files
	void iterate(Pixel imageDims,
				 RenderBuffer<Device::CUDA> outputBuffer) const;

	bool m_reset = true;
	ParameterHandler<PMinPathLength, PMaxPathLength, PNeeCount, PNeePositionGuide> m_params;
	scene::SceneHandle m_currentScene = nullptr;
	scene::SceneDescriptor<Device::CUDA>* m_scenePtr = nullptr;
};

}} // namespace mufflon::renderer