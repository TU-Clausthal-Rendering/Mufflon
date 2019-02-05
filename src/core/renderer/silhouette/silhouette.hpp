#pragma once

#include "core/math/rng.hpp"
#include "core/memory/residency.hpp"
#include "core/renderer/renderer.hpp"
#include "core/scene/descriptors.hpp"
#include <vector>

namespace mufflon::renderer {

template < Device >
struct RenderBuffer;

class SilhouetteTracer : public IRenderer {
public:
	// Initialize all resources required by this renderer.
	SilhouetteTracer();
	~SilhouetteTracer() = default;

	virtual void iterate(OutputHandler& outputBuffer) override;
	virtual void reset() override;
	virtual IParameterHandler& get_parameters() final { return m_params; }
	virtual bool has_scene() const noexcept override { return m_currentScene != nullptr; }
	virtual void load_scene(scene::SceneHandle scene, const ei::IVec2& resolution) override;
	virtual StringView get_name() const noexcept { return "Shadow silhouettes"; }
	virtual bool uses_device(Device dev) noexcept override { return may_use_device(dev); }
	static bool may_use_device(Device dev) noexcept { return Device::CPU == dev; }
private:
	// Create one sample path (actual PT algorithm)
	void sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer,
				const scene::SceneDescriptor<Device::CPU>& scene);
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	bool m_reset = true;
	ParameterHandler<PMinPathLength, PMaxPathLength, PNeeCount, PNeePositionGuide> m_params = {};
	scene::SceneHandle m_currentScene = nullptr;
	std::vector<math::Rng> m_rngs;
	scene::SceneDescriptor<Device::CPU> m_sceneDesc;
};

} // namespace mufflon::renderer