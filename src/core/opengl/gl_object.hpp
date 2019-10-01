#pragma once
#include "gl_wrapper.hpp"
#include <algorithm>

namespace mufflon::gl {
    
template < class Deleter >
class GlObject {
public:
    // default ctor: empty handle
	GlObject() = default;
    explicit GlObject(gl::Handle handle) noexcept : m_handle(handle) {}
    // forbid object copy
	GlObject(const GlObject&) = delete;
	GlObject& operator=(const GlObject&) = delete;
    // enable object move
	GlObject(GlObject&& o) noexcept : m_handle(o.m_handle) { o.m_handle = 0; }
    GlObject& operator=(GlObject&& o) noexcept {
		std::swap(m_handle, o.m_handle);
		return *this;
	}
	~GlObject() {
		reset();
	}
    // delete handle
    void reset() noexcept {
		Deleter::del(m_handle);
		m_handle = 0;
    }
	operator gl::Handle() const noexcept { return m_handle; }
    // reset handle and give address
    gl::Handle* operator&() noexcept {  // NOLINT(google-runtime-operator)
		reset();
		return &m_handle;
	}
private:
	gl::Handle m_handle = 0;
};

namespace detail {
    struct TextureDeleter {
		static void del(gl::Handle handle) noexcept;
    };
    struct BufferDeleter {
		static void del(gl::Handle handle) noexcept;
    };
    struct FramebufferDeleter {
		static void del(gl::Handle handle) noexcept;
    };
    struct ShaderDeleter {
		static void del(gl::Handle handle) noexcept;
    };
    struct ProgramDeleter {
		static void del(gl::Handle handle) noexcept;
    };
    struct VertexArrayDeleter {
		static void del(gl::Handle handle) noexcept;
    };
    struct SamplerDeleter {
		static void del(gl::Handle handle) noexcept;
    };
} // namespace detail

// handles that automatically delete themselves on destruction
using Texture = GlObject<detail::TextureDeleter>;
using Buffer = GlObject<detail::BufferDeleter>;
using Framebuffer = GlObject<detail::FramebufferDeleter>;
using Shader = GlObject<detail::ShaderDeleter>;
using Program = GlObject<detail::ProgramDeleter>;
using VertexArray = GlObject<detail::VertexArrayDeleter>;
using Sampler = GlObject<detail::SamplerDeleter>;
}