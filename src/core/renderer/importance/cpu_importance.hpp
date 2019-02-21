#pragma once

#include "importance_params.hpp"
#include "core/math/rng.hpp"
#include "core/memory/allocator.hpp"
#include "core/memory/residency.hpp"
#include "core/renderer/renderer.hpp"
#include "core/scene/descriptors.hpp"
#include <OpenMesh/Core/Utils/Property.hh>
#include <atomic>
#include <vector>

// Decimation according to 'Illumination-driven Mesh Reduction for Accelerating Light Transport Simulations' (Andreas Reich, 2015)

namespace mufflon::renderer {

template < Device >
struct RenderBuffer;

template < typename T, int A >
class PathVertex;

class CpuImportanceDecimater : public IRenderer {
public:
	// Initialize all resources required by this renderer.
	CpuImportanceDecimater();
	~CpuImportanceDecimater() = default;

	virtual void iterate(OutputHandler& outputBuffer) override;
	virtual void reset() override;
	virtual IParameterHandler& get_parameters() final { return m_params; }
	virtual bool has_scene() const noexcept override { return m_currentScene != nullptr; }
	virtual void load_scene(scene::SceneHandle scene, const ei::IVec2& resolution) override;
	virtual StringView get_name() const noexcept { return "Importance decimation"; }
	virtual bool uses_device(Device dev) noexcept override { return may_use_device(dev); }
	static bool may_use_device(Device dev) noexcept { return Device::CPU == dev; }

private:
	using PtPathVertex = PathVertex<u8, 4>;

	// Create one sample path (actual PT algorithm)
	void pt_sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer,
				   const scene::SceneDescriptor<Device::CPU>& scene);
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	void importance_sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer,
						   const scene::SceneDescriptor<Device::CPU>& scene);

	void initialize_importance_map();
	void gather_importance(RenderBuffer<Device::CPU>& buffer);
	bool trace_shadow_silhouette(const ei::Ray& shadowRay, const PtPathVertex& vertex,
								 const float lightDist, const float importance);
	bool trace_shadow_silhouette_shadow(const ei::Ray& shadowRay, const PtPathVertex& vertex,
										const scene::PrimitiveHandle& firstHit,
										const float lightDist, const float firstHitT,
										const float importance);
	void decimate(const ei::IVec2& resolution);
	void compute_max_importance();
	void display_importance(RenderBuffer<Device::CPU>& buffer);
	float compute_importance(const scene::PrimitiveHandle& hitId);

	bool m_reset = true;
	ImportanceParameters m_params = {};
	scene::SceneHandle m_currentScene = nullptr;
	std::vector<math::Rng> m_rngs;
	scene::SceneDescriptor<Device::CPU> m_sceneDesc;

	// Data buffer for importance
	unique_device_ptr<Device::CPU, std::atomic<float>[]> m_importanceMap;
	// Data buffer for vertex offset per instance for quick lookup
	unique_device_ptr<Device::CPU, u32[]> m_vertexOffsets;
	u32 m_vertexCount = 0;

	// Superfluous
	bool m_gotImportance = false;
	bool m_finishedDecimation = false;
	u32 m_currentImportanceIteration = 0u;
	float m_maxImportance;
};

} // namespace mufflon::renderer