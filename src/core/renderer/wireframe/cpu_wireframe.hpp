#pragma once

#include "wireframe_params.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/renderer_base.hpp"
#include <vector>

namespace mufflon::renderer {

template < Device >
struct RenderBuffer;

class CpuWireframe final : public RendererBase<Device::CPU> {
public:
	// Initialize all resources required by this renderer.
	CpuWireframe();
	~CpuWireframe() = default;

	void iterate() final;
	IParameterHandler& get_parameters() final { return m_params; }
	StringView get_name() const noexcept final { return "Wireframe"; }
	StringView get_short_name() const noexcept final { return "WF"; }

	void post_descriptor_requery() final;

private:
	// Create one sample path (actual PT algorithm)
	void sample(const Pixel coord);
	// Reset the initialization of the RNGs. If necessary also changes the number of RNGs.
	void init_rngs(int num);

	ei::Vec3 translateToWorldSpace(const ei::Vec3& point, const i32 instanceId) const;
	ei::Vec3 computeClosestLinePoint(const i32 instanceId, const ei::IVec3& indices, const ei::Vec3& hitpoint) const;	// Triangle
	ei::Vec3 computeClosestLinePoint(const i32 instanceId, const ei::IVec4& indices, const ei::Vec3& hitpoint) const;	// Quad
	ei::Vec3 computeClosestLinePoint(const i32 instanceId, const u32 index, const ei::Vec3& hitpoint,
									 const ei::Vec3& incident) const;			// Sphere

	WireframeParameters m_params = {};
	std::vector<math::Rng> m_rngs;
};

} // namespace mufflon::renderer