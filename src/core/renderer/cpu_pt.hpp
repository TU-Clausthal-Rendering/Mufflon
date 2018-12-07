#pragma once

#include "renderer.hpp"
#include "core/scene/scene.hpp"
#include "core/math/rng.hpp"
#include "core/cameras/camera_sampling.hpp"
#include <vector>

namespace mufflon::cameras {
	struct CameraParams;
} // namespace mufflon::cameras

namespace mufflon::renderer {

template < Device >
struct RenderBuffer;

class CpuPathTracer : public IRenderer {
public:
	// Initialize all resources required by this renderer.
	CpuPathTracer(scene::SceneHandle scene);

	virtual void iterate(OutputHandler& outputBuffer) override;
	virtual void reset() override;
	virtual IParameterHandler& get_parameters() final { return m_params; }
private:
	// Create one sample path (actual PT algorithm)
	void sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer,
				const scene::lights::LightTree<Device::CPU>& lightTree);
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);
	const cameras::CameraParams& get_cam() const {
		return *as<cameras::CameraParams>(m_camParams);
	}

	bool m_reset = true;
	ParameterHandler<PMinPathLength, PMaxPathLength, PNeeCount, PNeePositionGuide> m_params;
	scene::SceneHandle m_currentScene;
	std::vector<math::Xoroshiro128> m_rngs;
	u8 m_camParams[cameras::MAX_CAMERA_PARAM_SIZE];
	scene::SceneDescriptor<Device::CPU> m_sceneDesc;
};

} // namespace mufflon::renderer