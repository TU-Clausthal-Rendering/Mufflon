#pragma once
#include "core/renderer/gl_renderer_base.h"
#include "debug_bvh_params.hpp"
#include "core/opengl/gl_pipeline.h"
#include "core/renderer/forward/dynamic_fragment_buffer.hpp"
#include "core/renderer/forward/box_pipeline.hpp"

namespace mufflon::renderer {
	class DebugBvhRenderer final : public GlRendererBase<DebugBvhTargets> {
	public:
		DebugBvhRenderer();
		~DebugBvhRenderer() override = default;

		void iterate() final;
		IParameterHandler& get_parameters() final { return m_params; }
		static constexpr StringView get_name_static() noexcept { return "DebugBvh"; }
		static constexpr StringView get_short_name_static() noexcept { return "BVH"; }
		StringView get_name() const noexcept final { return get_name_static(); }
		StringView get_short_name() const noexcept final { return get_short_name_static(); }

		void post_reset() override;

	private:
		void init();
		static void upload_box_array(const scene::AccelDescriptor& accel, gl::Buffer& dstBuffer, int& boxCount, const ei::Box& root);

		DebugBvhParameters m_params = {};

		gl::Program m_triangleProgram;
		gl::Program m_quadProgram;
		gl::Program m_sphereProgram;

		gl::Pipeline m_trianglePipe;
		gl::Pipeline m_quadPipe;
		gl::Pipeline m_spherePipe;

		gl::VertexArray m_triangleVao;
		gl::VertexArray m_spheresVao;

		gl::Buffer m_transformBuffer;

		// transparency stuff
		BoxPipeline m_boxPipe;
		DynamicFragmentBuffer m_dynFragmentBuffer;
		int m_botLevelNumBoxes = 0;
		gl::Buffer m_botLevelBoxes;

		int m_topLevelNumBoxes = 0;
		gl::Buffer m_topLevelBoxes;

		gl::Buffer m_bboxes;

		bool m_showBoxes = false;
		bool m_showTopLevel = false;
		bool m_showBotLevel = false;
		int m_botIdx = -1;

		// init tracking
		bool m_isInit = false;
	};
}
