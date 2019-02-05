#include "interface.h"
#include "opengldisplay/opengl/program.hpp"
#include "opengldisplay/opengl/shader.hpp"
#include "opengldisplay/opengl/texture.hpp"
#include "opengldisplay/opengl/vao.hpp"
// Undefine windows header macros
#undef ERROR
#include "util/log.hpp"
#include <memory>
#include <string>


using namespace opengl;
using namespace mufflon;

#define FUNCTION_NAME __func__

namespace {

constexpr const char* VERTEX_CODE = "#version 330 core\nvoid main(){}";
constexpr const char* GEOMETRY_CODE =
	"#version 330 core\n"
	"layout(points) in;"
	"layout(triangle_strip, max_vertices = 4) out;"
	"uniform vec4 uvs;"
	"out vec2 texcoord;"
	"void main() {"
	"	gl_Position = vec4(1.0, 1.0, 0.0, 1.0);"
	"	texcoord = vec2(uvs.z, uvs.w);"
	"	EmitVertex();"
	"	gl_Position = vec4(-1.0, 1.0, 0.0, 1.0);"
	"	texcoord = vec2(uvs.x, uvs.w);"
	"	EmitVertex();"
	"	gl_Position = vec4(1.0, -1.0, 0.0, 1.0);"
	"	texcoord = vec2(uvs.z, uvs.y);"
	"	EmitVertex();"
	"	gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);"
	"	texcoord = vec2(uvs.x, uvs.y);"
	"	EmitVertex();"
	"	EndPrimitive();"
	"}";
constexpr const char* FRAGMENT_CODE =
	"#version 330 core\n"
	"in vec2 texcoord;"
	"uniform sampler2D textureSampler;"
	"uniform float gamma;"
	"uniform float factor;"
	"void main() {"
	"	vec3 rgb = texture2D(textureSampler, texcoord).rgb;"
	"	float luminance = 0.299*rgb.r + 0.587*rgb.g + 0.114*rgb.b;"
	"	gl_FragColor.xyz = factor * rgb / luminance * pow(luminance, 1.0/gamma);"
	"}";

std::unique_ptr<Program> s_screenProgram = nullptr;
float s_gamma = 1.0f;
float s_factor = 1.f;
GLint s_textureSamplerUniform;
GLint s_gammaUniform;
GLint s_factorUniform;
GLint s_uvUniform;
std::unique_ptr<VertexArray> s_vao = nullptr;
std::unique_ptr<Texture2D> s_screenTexture = nullptr;
std::string s_lastError;
static void(*s_logCallback)(const char*, int);


// Function delegating the logger output to the applications handle, if applicable
void delegateLog(LogSeverity severity, const std::string& message) {
	// Try-catching here is pointless since we have no way of communicating failure
	if(s_logCallback != nullptr)
		s_logCallback(message.c_str(), static_cast<int>(severity));
	if(severity == LogSeverity::ERROR || severity == LogSeverity::FATAL_ERROR) {
		s_lastError = message;
	}
}

void APIENTRY opengl_callback(GLenum source, GLenum type, GLuint id,
							  GLenum severity, GLsizei length,
							  const GLchar* message, const void* userParam) {
	switch(severity) {
		case GL_DEBUG_SEVERITY_HIGH: logError(message); break;
		case GL_DEBUG_SEVERITY_MEDIUM: logWarning(message); break;
		case GL_DEBUG_SEVERITY_LOW: logInfo(message); break;
		default: logPedantic(message); break;
	}
}

} // namespace

void opengldisplay_set_gamma(float val) {
	s_gamma = val;
}

float opengldisplay_get_gamma() {
	return s_gamma;
}

void opengldisplay_set_factor(float val) {
	s_factor = val;
}

float opengldisplay_get_factor() {
	return s_factor;
}

const char* opengldisplay_get_dll_error() {
	return s_lastError.c_str();
}

Boolean opengldisplay_display(int left, int right, int bottom, int top, uint32_t width, uint32_t height) {
	try {
		s_screenProgram->activate();
		s_vao->bind();
		s_screenTexture->bind_as_texture(0u);

		// Set the viewport size
		glViewport(0, 0, static_cast<GLsizei>(right - left), static_cast<GLsizei>(top - bottom));
		// Also enable sRgb
		glEnable(GL_FRAMEBUFFER_SRGB);
		// Set the texture coordinates of the fullscreen quad to emulate scissoring
		::glUniform4f(s_uvUniform,
					  static_cast<float>(left) / static_cast<float>(width),
					  static_cast<float>(bottom) / static_cast<float>(height),
					  static_cast<float>(right) / static_cast<float>(width),
					  static_cast<float>(top) / static_cast<float>(height));
		::glUniform1i(s_textureSamplerUniform, 0);
		::glUniform1f(s_gammaUniform, s_gamma);
		::glUniform1f(s_factorUniform, s_factor);

		::glDrawArrays(GL_POINTS, 0, 1u);
	} catch(const std::exception& e) {
		logError("[", FUNCTION_NAME, "] Caught exception: ", e.what());
		return false;
	}
	return true;
}

Boolean opengldisplay_resize_screen(uint32_t width, uint32_t height, TextureFormat format) {
	// TODO: check format?
	try {
		s_screenTexture = std::make_unique<Texture2D>();
		Texture2D::InternalFormat internalFormat = Texture2D::InternalFormat::INVALID;
		switch(format) {
			case TextureFormat::FORMAT_R8U: internalFormat = Texture2D::InternalFormat::R8UI; break;
			case TextureFormat::FORMAT_RG8U: internalFormat = Texture2D::InternalFormat::RG8UI; break;
			case TextureFormat::FORMAT_RGBA8U: internalFormat = Texture2D::InternalFormat::RGBA8UI; break;
			case TextureFormat::FORMAT_R16U: internalFormat = Texture2D::InternalFormat::R16UI; break;
			case TextureFormat::FORMAT_RG16U: internalFormat = Texture2D::InternalFormat::RG16UI; break;
			case TextureFormat::FORMAT_RGBA16U: internalFormat = Texture2D::InternalFormat::RGBA16UI; break;
			case TextureFormat::FORMAT_R16F: internalFormat = Texture2D::InternalFormat::R16F; break;
			case TextureFormat::FORMAT_RG16F: internalFormat = Texture2D::InternalFormat::RG16F; break;
			case TextureFormat::FORMAT_RGBA16F: internalFormat = Texture2D::InternalFormat::RGBA16F; break;
			case TextureFormat::FORMAT_R32F: internalFormat = Texture2D::InternalFormat::R32F; break;
			case TextureFormat::FORMAT_RG32F: internalFormat = Texture2D::InternalFormat::RG32F; break;
			case TextureFormat::FORMAT_RGBA32F: internalFormat = Texture2D::InternalFormat::RGBA32F; break;
		};

		s_screenTexture->allocate_storage(static_cast<GLsizei>(width), static_cast<GLsizei>(height),
										  internalFormat);
	} catch(const std::exception& e) {
		logError("[", FUNCTION_NAME, "] Caught exception: ", e.what());
		return false;
	}
	return true;
}

Boolean opengldisplay_write(const char* data) {
	if(data == nullptr) {
		logError("[", FUNCTION_NAME, "] Data is nullptr");
	}
	// Determine the pixel format for OpenGL
	GLenum type = GL_INVALID_ENUM;

	// Since we know what we came from, we can induce the pixel format from the internal format
	switch(s_screenTexture->get_format()) {
		case Texture2D::InternalFormat::R8UI:
		case Texture2D::InternalFormat::RG8UI:
		case Texture2D::InternalFormat::RGBA8UI:
			type = GL_UNSIGNED_BYTE;
			break;
		case Texture2D::InternalFormat::R16UI:
		case Texture2D::InternalFormat::RG16UI:
		case Texture2D::InternalFormat::RGBA16UI:
			type = GL_UNSIGNED_SHORT;
			break;
		case Texture2D::InternalFormat::R16F:
		case Texture2D::InternalFormat::RG16F:
		case Texture2D::InternalFormat::RGBA16F:
			type = GL_HALF_FLOAT;
			break;
		case Texture2D::InternalFormat::R32F:
		case Texture2D::InternalFormat::RG32F:
		case Texture2D::InternalFormat::RGBA32F:
			type = GL_FLOAT;
			break;
		default:
			logError("[", FUNCTION_NAME, "] Output buffer has unknown format!");
			return false;
	}

	s_screenTexture->bind_as_texture(0u);
	::glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 800, 600, 0, GL_RGBA, GL_FLOAT, data);
	/*::glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(s_screenTexture->get_width()),
					  static_cast<GLsizei>(s_screenTexture->get_height()),
					  static_cast<GLenum>(s_screenTexture->get_format()), type,
					  data);*/
	return true;
}


Boolean opengldisplay_initialize() {
	static bool initialized = false;

	if(!initialized) {
		try {
			if(!gladLoadGL()) {
				logError("[", FUNCTION_NAME, "] gladLoadGL failed");
				return false;
			}

			glDebugMessageCallback(opengl_callback, nullptr);
			glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

			Shader vertex{ Shader::Type::VERTEX_SHADER };
			Shader geometry{ Shader::Type::GEOMETRY_SHADER };
			Shader fragment{ Shader::Type::FRAGMENT_SHADER };
			vertex.attach_source(VERTEX_CODE);
			geometry.attach_source(GEOMETRY_CODE);
			fragment.attach_source(FRAGMENT_CODE);
			if(!vertex.compile()) {
				logError("[", FUNCTION_NAME, "] Failed to compile vertex shader: ", vertex.get_info_log());
				return false;
			}
			if(!geometry.compile()) {
				logError("[", FUNCTION_NAME, "] Failed to compile geometry shader: ", geometry.get_info_log());
				return false;
			}
			if(!fragment.compile()) {
				logError("[", FUNCTION_NAME, "] Failed to compile fragment shader: ", fragment.get_info_log());
				return false;
			}

			s_screenProgram = std::make_unique<Program>();
			s_screenProgram->attach(std::move(vertex));
			s_screenProgram->attach(std::move(geometry));
			s_screenProgram->attach(std::move(fragment));
			if(!s_screenProgram->link()) {
				s_lastError = "[initialize] Failed to link program: " + s_screenProgram->link();
				return false;
			}
			s_screenProgram->activate();

			s_textureSamplerUniform = s_screenProgram->get_uniform_location("textureSampler");
			s_gammaUniform = s_screenProgram->get_uniform_location("gamma");
			s_factorUniform = s_screenProgram->get_uniform_location("factor");
			s_uvUniform = s_screenProgram->get_uniform_location("uvs");


			s_vao = std::make_unique<VertexArray>();
			s_screenTexture = std::make_unique<Texture2D>();

			initialized = true;
		} catch(const std::exception& e) {
			logError("[", FUNCTION_NAME, "] Caught exception: ", e.what());
			s_screenProgram->detach_all();
			return false;
		}
	}

	return true;
}

Boolean opengldisplay_set_logger(void(*logCallback)(const char*, int)) {
	try {
		if(s_logCallback == nullptr) {
			registerMessageHandler(delegateLog);
			disableStdHandler();
		}
		s_logCallback = logCallback;
		return true;
	} catch(const std::exception& e) {
		logError("[", FUNCTION_NAME, "] Caught exception: ", e.what());
		s_screenProgram->detach_all();
		return false;
	}
}

void opengldisplay_destroy() {
	s_screenProgram.reset();
	s_vao.reset();
	s_screenTexture.reset();
}

Boolean opengldisplay_set_log_level(LogLevel level) {

	switch(level) {
		case LogLevel::LOG_PEDANTIC:
			mufflon::s_logLevel = LogSeverity::PEDANTIC;
			return true;
		case LogLevel::LOG_INFO:
			mufflon::s_logLevel = LogSeverity::INFO;
			return true;
		case LogLevel::LOG_WARNING:
			mufflon::s_logLevel = LogSeverity::WARNING;
			return true;
		case LogLevel::LOG_ERROR:
			mufflon::s_logLevel = LogSeverity::ERROR;
			return true;
		case LogLevel::LOG_FATAL_ERROR:
			mufflon::s_logLevel = LogSeverity::FATAL_ERROR;
			return true;
		default:
			logError("[", FUNCTION_NAME, "] Invalid log level");
			return false;
	}
}