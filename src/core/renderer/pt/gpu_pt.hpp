#pragma once

#include "pt_params.hpp"
#include "core/renderer/renderer.hpp"
#include "core/scene/descriptors.hpp"
#include "core/scene/handles.hpp"
#include "core/scene/lights/light_tree.hpp"

namespace mufflon {

// Forward declarations
enum class Device : unsigned char;

namespace renderer {

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
	virtual StringView get_name() const noexcept { return "Pathtracer"; }
	virtual bool uses_device(Device dev) noexcept override { return may_use_device(dev); }
	static bool may_use_device(Device dev) noexcept { return Device::CUDA == dev; }

private:
	// Used so that we don't need to include everything in CU files
	void iterate(Pixel imageDims,
				 RenderBuffer<Device::CUDA> outputBuffer) const;

	bool m_reset = true;
	PtParameters m_params;
	scene::SceneHandle m_currentScene = nullptr;
	scene::SceneDescriptor<Device::CUDA>* m_scenePtr = nullptr;
};

}} // namespace mufflon::renderer