#pragma once

#include <glad/glad.h>
#include <stdexcept>

namespace opengl {

/* Veeeery simple texture wrapper (lacking tons of features that we don't 
 * for showing a texture on-screen).
 */
class Texture2D {
public:
	enum class InternalFormat : GLint {
		INVALID = GL_INVALID_ENUM,

		// One channel
		R8 = GL_R8,
		R8S = GL_R8_SNORM,
		R8I = GL_R8I,
		R8UI = GL_R8UI,
		R16 = GL_R16,
		R16S = GL_R16_SNORM,
		R16I = GL_R16I,
		R16UI = GL_R16UI,
		R16F = GL_R16F,
		R32I = GL_R32I,
		R32UI = GL_R32UI,
		R32F = GL_R32F,

		// Two channels
		RG8 = GL_RG8,
		RG8S = GL_RG8_SNORM,
		RG8I = GL_RG8I,
		RG8UI = GL_RG8UI,
		RG16 = GL_RG16,
		RG16S = GL_RG16_SNORM,
		RG16I = GL_RG16I,
		RG16UI = GL_RG16UI,
		RG16F = GL_RG16F,
		RG32I = GL_RG32I,
		RG32UI = GL_RG32UI,
		RG32F = GL_RG32F,

		// Three channels
		R3_G3_B2 = GL_R3_G3_B2,
		RGB4 = GL_RGB4,
		RGB5 = GL_RGB5,
		RGB8 = GL_RGB8,
		RGB8S = GL_RGB8_SNORM,
		RGB8I = GL_RGB8I,
		RGB8UI = GL_RGB8UI,
		SRGB8 = GL_SRGB8,
		RGB10 = GL_RGB10,
		RGB12 = GL_RGB12,
		RGB16 = GL_RGB16,
		RGB16S = GL_RGB16_SNORM,
		RGB16I = GL_RGB16I,
		RGB16UI = GL_RGB16UI,
		RGB16F = GL_RGB16F,
		RGB32I = GL_RGB32I,
		RGB32UI = GL_RGB32UI,
		RGB32F = GL_RGB32F,
		R11F_G11F_B10F = GL_R11F_G11F_B10F,
		RGB9_E5 = GL_RGB9_E5,

		// Four channels
		RGBA2 = GL_RGBA2,
		RGBA4 = GL_RGBA4,
		RGB5_A1 = GL_RGB5_A1,
		RGBA8 = GL_RGBA8,
		RGBA8S = GL_RGBA8_SNORM,
		RGBA8I = GL_RGBA8I,
		RGBA8UI = GL_RGBA8UI,
		SRGB8_ALPHA8 = GL_SRGB8_ALPHA8,
		RGB10_A2 = GL_RGB10_A2,
		RGB10_A2UI = GL_RGB10_A2UI,
		RGBA12 = GL_RGBA12,
		RGBA16 = GL_RGBA16,
		RGBA16S = GL_RGBA16_SNORM,
		RGBA16I = GL_RGBA16I,
		RGBA16UI = GL_RGBA16UI,
		RGBA16F = GL_RGBA16F,
		RGBA32I = GL_RGBA32I,
		RGBA32UI = GL_RGBA32UI,
		RGBA32F = GL_RGBA32F,

		// Depth stencil formats
		DEPTH_COMPONENT32F = GL_DEPTH_COMPONENT32F,
		DEPTH_COMPONENT24 = GL_DEPTH_COMPONENT24,
		DEPTH_COMPONENT16 = GL_DEPTH_COMPONENT16,
		DEPTH32F_STENCIL8 = GL_DEPTH32F_STENCIL8,
		DEPTH24_STENCIL8 = GL_DEPTH24_STENCIL8,
		STENCIL_INDEX8 = GL_STENCIL_INDEX8,
	};

	enum class PixelFormat : GLenum {
		RED = GL_RED,
		RG = GL_RG,
		RGB = GL_RGB,
		BGR = GL_BGR,
		RGBA = GL_RGBA,
		BGRA = GL_BGRA,
		RED_INTEGER = GL_RED_INTEGER,
		RG_INTEGER = GL_RG_INTEGER,
		RGB_INTEGER = GL_RGB_INTEGER,
		BGR_INTEGER = GL_BGR_INTEGER,
		RGBA_INTEGER = GL_RGBA_INTEGER,
		BGRA_INTEGER = GL_BGRA_INTEGER,
		STENCIL_INDEX = GL_STENCIL_INDEX,
		DEPTH_COMPONENT = GL_DEPTH_COMPONENT,
		DEPTH_STENCIL = GL_DEPTH_STENCIL
	};

	enum class PixelType : GLenum {
		UNSIGNED_BYTE = GL_UNSIGNED_BYTE,
		BYTE = GL_BYTE,
		UNSIGNED_SHORT = GL_UNSIGNED_SHORT,
		SHORT = GL_SHORT,
		UNSIGNED_INT = GL_UNSIGNED_INT,
		INT = GL_INT,
		HALF_FLOAT = GL_HALF_FLOAT,
		FLOAT = GL_FLOAT,
		INVALID = GL_INVALID_ENUM
	};

	Texture2D() :
		m_id(0),
		m_format(InternalFormat::INVALID),
		m_width(0),
		m_height(0)
	{
		::glGenTextures(1u, &m_id);
		if(m_id == 0u)
			throw std::runtime_error("Failed to create texture object");
		bind_as_texture(0u);
		::glTextureParameteri(m_id, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		::glTextureParameteri(m_id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		::glTextureParameteri(m_id, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		::glTextureParameteri(m_id, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	void allocate_storage(GLsizei width, GLsizei height, InternalFormat format)	{
		if(m_id == 0u)
			throw std::runtime_error("Invalid texture object");
		m_width = width;
		m_height = height;
		m_format = format;
		::glTextureStorage2D(m_id, 1u, static_cast<GLenum>(m_format), m_width, m_height);
	}
	Texture2D(const Texture2D&) = delete;
	Texture2D(Texture2D&& tex) : m_id(tex.m_id) {
		tex.m_id = 0u;
	}
	Texture2D& operator=(const Texture2D&) = delete;
	Texture2D& operator=(Texture2D&& tex) {
		std::swap(m_id, tex.m_id);
		return *this;
	}
	~Texture2D() {
		if(m_id != 0u)
			::glDeleteTextures(1u, &m_id);
	}

	void bind_as_texture() {
		glBindTexture(GL_TEXTURE_2D, m_id);
	}
	void bind_as_texture(GLuint textureUnitIndex) {
		glActiveTexture(GL_TEXTURE0 + textureUnitIndex);
		glBindTexture(GL_TEXTURE_2D, m_id);
	}

	static void unbind_texture() {
		glBindTexture(GL_TEXTURE_2D, 0u);
	}
	static void unbind_texture(GLuint textureUnitIndex) {
		glActiveTexture(GL_TEXTURE0 + textureUnitIndex);
		glBindTexture(GL_TEXTURE_2D, 0u);
	}

	static GLuint get_active_texture_unit() {
		GLint unit = 0;
		::glGetIntegerv(GL_TEXTURE_BINDING_2D, &unit);
		return static_cast<GLuint>(unit - GL_TEXTURE0);
	}

	void set_data(PixelFormat format, PixelType type, void* data) {
		this->bind_as_texture();
		if(m_id == 0)
			throw std::runtime_error("Invalid texture object");
		::glTexImage2D(GL_TEXTURE_2D, 0u, static_cast<GLint>(m_format), m_width, m_height,
					   0, static_cast<GLenum>(format), static_cast<GLenum>(type), data);
	}

	GLuint get_handle() const noexcept { return m_id; }
	GLuint get_width() const noexcept { return m_width; }
	GLuint get_height() const noexcept { return m_height; }
	InternalFormat get_format() const noexcept { return m_format; }

private:
	GLuint m_id;
	GLsizei m_width;
	GLsizei m_height;
	InternalFormat m_format;
};

} // namespace opengl