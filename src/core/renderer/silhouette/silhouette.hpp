#pragma once

#include "core/math/rng.hpp"
#include "core/memory/residency.hpp"
#include "core/renderer/renderer.hpp"
#include "core/scene/descriptors.hpp"
#include <vector>

namespace mufflon::renderer::silhouette {

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
	virtual std::string_view get_name() const noexcept { return "Shadow silhouettes"; }
	static bool uses_device(Device dev) noexcept { return Device::CPU == dev; }
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

} // namespace mufflon::renderer::silhouette