#include "interface.h"
#include "plugin/texture_plugin.hpp"
#include "util/log.hpp"
#include "util/byte_io.hpp"
#include "util/punning.hpp"
#include "util/degrad.hpp"
#include "ei/vector.hpp"
#include "profiler/cpu_profiler.hpp"
#include "profiler/gpu_profiler.hpp"
#include "core/renderer/renderer.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/cpu_pt.hpp"
#include "core/renderer/gpu_pt.hpp"
#include "core/cameras/pinhole.hpp"
#include "core/scene/object.hpp"
#include "core/scene/world_container.hpp"
#include "core/scene/geometry/polygon.hpp"
#include "core/scene/geometry/sphere.hpp"
#include "core/scene/lights/lights.hpp"
#include "core/scene/materials/lambert.hpp"
#include "mffloader/interface/interface.h"
#include <cuda_runtime.h>
#include <type_traits>
#include <mutex>
#include <fstream>
#include <vector>
// TODO: remove this (leftover from Felix' prototype)
#include <glad/glad.h>

#ifdef _WIN32
#include <minwindef.h>
#include <combaseapi.h>
#else // _WIN32
#include <dlfcn.h>
#endif // _WIN32

// Undefine unnecessary windows macros
#undef near
#undef far
#undef ERROR

using namespace mufflon;
using namespace mufflon::scene;
using namespace mufflon::scene::geometry;

// Helper macros for error checking and logging
#define FUNCTION_NAME __func__
#define CHECK(x, name, retval)													\
	do {																		\
		if(!(x)) {																\
			logError("[", FUNCTION_NAME, "] Violated condition " #name " ("		\
			#x ")");															\
			return retval;														\
		}																		\
	} while(0)
#define CHECK_NULLPTR(x, name, retval)											\
	do {																		\
		if((x) == nullptr) {													\
			logError("[", FUNCTION_NAME, "] Invalid " #name " (nullptr)");		\
			return retval;														\
		}																		\
	} while(0)
#define CHECK_GEQ_ZERO(x, name, retval)											\
	do {																		\
		if((x) < 0) {															\
			logError("[", FUNCTION_NAME, "] Invalid " #name " (< 0)");			\
			return retval;														\
		}																		\
	} while(0)
#define TRY try {
#define CATCH_ALL(retval)														\
	} catch(const std::exception& e) {											\
		logError("[", FUNCTION_NAME, "] Exception caught: ", e.what());			\
		s_lastError = e.what();													\
		return retval;															\
	}

// Shortcuts for long handle names
using PolyVHdl = Polygons::VertexHandle;
using PolyFHdl = Polygons::FaceHandle;
using SphereVHdl = Spheres::SphereHandle;

// Return values for invalid handles/attributes
namespace {

// static variables for interacting with the renderer
std::unique_ptr<renderer::IRenderer> s_currentRenderer;
std::unique_ptr<renderer::OutputHandler> s_imageOutput;
renderer::OutputValue s_outputTargets{ 0 };
WorldContainer& s_world = WorldContainer::instance();
static void(*s_logCallback)(const char*, int);
// Holds the CUDA device index
int s_cudaDevIndex = -1;
// Holds the last error for the GUI to display
std::string s_lastError;

// Plugin container
std::vector<TextureLoaderPlugin> s_plugins;


constexpr PolygonAttributeHandle INVALID_POLY_VATTR_HANDLE{
	INVALID_INDEX,
	AttribDesc{
		AttributeType::ATTR_COUNT,
		0u
	},
	false
};
constexpr PolygonAttributeHandle INVALID_POLY_FATTR_HANDLE{
	INVALID_INDEX,
	AttribDesc{
		AttributeType::ATTR_COUNT,
		0u
	},
	true
};
constexpr SphereAttributeHandle INVALID_SPHERE_ATTR_HANDLE{
	INVALID_INDEX,
	AttribDesc{
		AttributeType::ATTR_COUNT,
		0u
	}
};

// Dummy type, passing another type into a lambda without needing
// to instantiate non-zero-sized data
template < class T >
struct TypeHolder {
	using Type = T;
};

// Helper functions to create the proper attribute type
template < class T, class L1, class L2 >
inline auto switchAttributeType(unsigned int rows, L1&& regular,
								L2&& noMatch) {
	switch(rows) {
		case 1u: return regular(TypeHolder<ei::Vec<T, 1u>>{});
		case 2u: return regular(TypeHolder<ei::Vec<T, 2u>>{});
		case 3u: return regular(TypeHolder<ei::Vec<T, 3u>>{});
		case 4u: return regular(TypeHolder<ei::Vec<T, 4u>>{});
		default: return noMatch();
	}
}
template < class L1, class L2 >
inline auto switchAttributeType(const AttribDesc& desc, L1&& regular,
								L2&& noMatch) {
	if(desc.rows == 1u) {
		switch(desc.type) {
			case AttributeType::ATTR_CHAR: return regular(TypeHolder<int8_t>{});
			case AttributeType::ATTR_UCHAR: return regular(TypeHolder<uint8_t>{});
			case AttributeType::ATTR_SHORT: return regular(TypeHolder<int16_t>{});
			case AttributeType::ATTR_USHORT: return regular(TypeHolder<uint16_t>{});
			case AttributeType::ATTR_INT: return regular(TypeHolder<int32_t>{});
			case AttributeType::ATTR_UINT: return regular(TypeHolder<uint32_t>{});
			case AttributeType::ATTR_LONG: return regular(TypeHolder<int64_t>{});
			case AttributeType::ATTR_ULONG: return regular(TypeHolder<uint64_t>{});
			case AttributeType::ATTR_FLOAT: return regular(TypeHolder<float>{});
			case AttributeType::ATTR_DOUBLE: return regular(TypeHolder<double>{});
			default: return noMatch();
		}
	} else {
		switch(desc.type) {
			case AttributeType::ATTR_UCHAR: return switchAttributeType<uint8_t>(desc.rows, std::move(regular), std::move(noMatch));
			case AttributeType::ATTR_INT: return switchAttributeType<int32_t>(desc.rows, std::move(regular), std::move(noMatch));
			case AttributeType::ATTR_FLOAT: return switchAttributeType<float>(desc.rows, std::move(regular), std::move(noMatch));
			default: return noMatch();
		}
	}
}

// Converts a polygon attribute to the C interface handle type
template < class AttrHdl >
inline AttrHdl convert_poly_to_attr(const PolygonAttributeHandle& hdl) {
	using OmAttrHandle = typename AttrHdl::OmAttrHandle;
	using CustomAttrHandle = typename AttrHdl::CustomAttrHandle;

	return AttrHdl{
		OmAttrHandle{static_cast<int>(hdl.openMeshIndex)},
		CustomAttrHandle{static_cast<size_t>(hdl.customIndex)}
	};
}

// Convert attribute type to string for convenience
inline std::string get_attr_type_name(AttribDesc desc) {
	std::string typeName;
	switch(desc.type) {
		case AttributeType::ATTR_CHAR: typeName = "char"; break;
		case AttributeType::ATTR_UCHAR: typeName = "uchar"; break;
		case AttributeType::ATTR_SHORT: typeName = "short"; break;
		case AttributeType::ATTR_USHORT: typeName = "ushort"; break;
		case AttributeType::ATTR_INT: typeName = "int"; break;
		case AttributeType::ATTR_UINT: typeName = "uint"; break;
		case AttributeType::ATTR_LONG: typeName = "long"; break;
		case AttributeType::ATTR_ULONG: typeName = "ulong"; break;
		case AttributeType::ATTR_FLOAT: typeName = "float"; break;
		case AttributeType::ATTR_DOUBLE: typeName = "double"; break;
		default: typeName = "unknown";
	}
	if(desc.rows != 1u)
		typeName = "Vec<" + typeName + ',' + std::to_string(desc.rows) + '>';
	return typeName;
}

// Function delegating the logger output to the applications handle, if applicable
void delegateLog(LogSeverity severity, const std::string& message) {
	TRY
	if(s_logCallback != nullptr)
		s_logCallback(message.c_str(), static_cast<int>(severity));
	if(severity == LogSeverity::ERROR || severity == LogSeverity::FATAL_ERROR) {
		s_lastError = message;
	}
	CATCH_ALL(;)
}


} // namespace

const char* core_get_dll_error() {
	TRY
	return s_lastError.c_str();
	CATCH_ALL(nullptr)
}

bool core_set_log_level(LogLevel level) {
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

Boolean display_screenshot() {
	TRY
	if(!s_imageOutput) {
		logError("[", FUNCTION_NAME, "] No image to print is present");
		return false;
	}

	constexpr float gamma = 2.2f;
	constexpr float exposure = 2.0f;

	constexpr const char* VERTEX_CODE = "#version 330 core\nvoid main(){}";
	constexpr const char* GEOMETRY_CODE =
		"#version 330 core\n"
		"layout(points) in;"
		"layout(triangle_strip, max_vertices = 4) out;"
		"out vec2 texcoord;"
		"void main() {"
		"	gl_Position = vec4(1.0, 1.0, 0.5, 1.0);"
		"	texcoord = vec2(1.0, 1.0);"
		"	EmitVertex();"
		"	gl_Position = vec4(-1.0, 1.0, 0.5, 1.0);"
		"	texcoord = vec2(0.0, 1.0);"
		"	EmitVertex();"
		"	gl_Position = vec4(1.0, -1.0, 0.5, 1.0);"
		"	texcoord = vec2(1.0, 0.0);"
		"	EmitVertex();"
		"	gl_Position = vec4(-1.0, -1.0, 0.5, 1.0);"
		"	texcoord = vec2(0.0, 0.0);"
		"	EmitVertex();"
		"	EndPrimitive();"
		"}";
	constexpr const char* FRAGMENT_CODE =
		"#version 330 core\n"
		"in vec2 texcoord;"
		"uniform sampler2D textureSampler;"
		"uniform float gamma;"
		"uniform float exposure;"
		"void main() {"
		"	vec3 rgb = texture2D(textureSampler, texcoord).rgb;"
		"	float luminance = 0.2126*rgb.r + 0.7252*rgb.g + 0.0722*rgb.b;"
		"	gl_FragColor.xyz = pow(vec3(1.f) - exp(-rgb * exposure), vec3(1.0 / gamma));"
		"}";

	GLuint tex = 0u;
	GLuint program = 0u;
	GLuint vertShader = 0u;
	GLuint geomShader = 0u;
	GLuint fragShader = 0u;
	GLuint vao = 0u;

	auto cleanup = [&tex, &vao, &program, &vertShader, &geomShader, &fragShader]() {
		glBindTexture(GL_TEXTURE_2D, 0u);
		glUseProgram(0);
		glBindVertexArray(0);
		if(tex != 0) glDeleteTextures(1u, &tex);
		if(vertShader != 0) glDeleteShader(vertShader);
		if(geomShader != 0) glDeleteShader(geomShader);
		if(fragShader != 0) glDeleteShader(fragShader);
		if(program != 0) glDeleteProgram(program);
		if(vao != 0) glDeleteVertexArrays(1, &vao);
	};

	// Creates an OpenGL texture, copies the screen data into it and
	// renders it to the screen
	glGenTextures(1u, &tex);
	if(tex == 0) {
		logError("[", FUNCTION_NAME, "] Failed to initialize screen texture");
		cleanup();
		return false;
	}
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	vertShader = glCreateShader(GL_VERTEX_SHADER);
	geomShader = glCreateShader(GL_GEOMETRY_SHADER);
	fragShader = glCreateShader(GL_FRAGMENT_SHADER);
	if(vertShader == 0 || fragShader == 0 || geomShader == 0) {
		logError("[", FUNCTION_NAME, "] Failed to initialize screen shaders");
		cleanup();
		return false;
	}
	glShaderSource(vertShader, 1u, &VERTEX_CODE, nullptr);
	glShaderSource(geomShader, 1u, &GEOMETRY_CODE, nullptr);
	glShaderSource(fragShader, 1u, &FRAGMENT_CODE, nullptr);
	glCompileShader(vertShader);
	glCompileShader(geomShader);
	glCompileShader(fragShader);
	GLint compiled[3];
	glGetShaderiv(vertShader, GL_COMPILE_STATUS, &compiled[0]);
	glGetShaderiv(geomShader, GL_COMPILE_STATUS, &compiled[1]);
	glGetShaderiv(fragShader, GL_COMPILE_STATUS, &compiled[2]);
	if(compiled[0] != GL_TRUE || compiled[1] != GL_TRUE || compiled[2] != GL_TRUE) {
		logError("[", FUNCTION_NAME, "] Failed to compile screen shaders!");

		GLint id;
		if(compiled[0] != GL_TRUE)
			id = vertShader;
		else if(compiled[1] != GL_TRUE)
			id = geomShader;
		else
			id = fragShader;
		GLint logLength;
		std::string msg;
		glGetShaderiv(id, GL_INFO_LOG_LENGTH, &logLength);
		msg.resize(logLength + 1);
		msg[logLength] = '\0';
		glGetShaderInfoLog(id, logLength, NULL, msg.data());
		logError("Error: ", msg);

		cleanup();
		return false;
	}

	program = glCreateProgram();
	if(program == 0u) {
		logError("[", FUNCTION_NAME, "] Failed to initialize screen program");
		cleanup();
		return false;
	}
	glAttachShader(program, vertShader);
	glAttachShader(program, geomShader);
	glAttachShader(program, fragShader);
	glLinkProgram(program);
	GLint linkStatus;
	glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
	if(linkStatus != GL_TRUE) {
		logError("[", FUNCTION_NAME, "] Failed to link screen program");
		cleanup();
		return false;
	}
	glDetachShader(program, vertShader);
	glDetachShader(program, geomShader);
	glDetachShader(program, fragShader);

	ei::IVec2 res = s_imageOutput->get_resolution();
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, res.x, res.y, 0, GL_RGBA, GL_FLOAT,
		s_imageOutput->get_data(renderer::OutputValue{ renderer::OutputValue::RADIANCE },
			textures::Format::RGBA32F, false).data());

	glUseProgram(program);
	glUniform1i(glGetUniformLocation(program, "textureSampler"), 0);
	glUniform1f(glGetUniformLocation(program, "gamma"), gamma);
	glUniform1f(glGetUniformLocation(program, "exposure"), exposure);

	glGenVertexArrays(1, &vao);
	if(vao == 0) {
		logError("[", FUNCTION_NAME, "] Failed to link screen program");
		cleanup();
		return false;
	}
	glBindVertexArray(vao);
	glDrawArrays(GL_POINTS, 0, 1u);

	cleanup();
	return true;
	CATCH_ALL(false)
}
Boolean resize(int width, int height, int offsetX, int offsetY) {
	TRY
	// glViewport should not be called with width or height zero
	if(width == 0 || height == 0) return true;
	glViewport(0, 0, width, height);
	return true;
	CATCH_ALL(false)
}
void execute_command(const char* command) {
	TRY
	// TODO
	CATCH_ALL(;)
}

Boolean polygon_reserve(ObjectHdl obj, size_t vertices, size_t edges, size_t tris, size_t quads) {
	TRY
	CHECK_NULLPTR(obj, "object handle", false);
	static_cast<Object*>(obj)->template get_geometry<Polygons>().reserve(vertices, edges, tris, quads);
	return true;
	CATCH_ALL(false)
}

PolygonAttributeHandle polygon_request_vertex_attribute(ObjectHdl obj, const char* name,
														AttribDesc type) {
	TRY
	CHECK_NULLPTR(obj, "object handle", INVALID_POLY_VATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_POLY_VATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [name, &type, &object](const auto& val) {
			using Type = typename std::decay_t<decltype(val)>::Type;
		auto attr = object.template get_geometry<Polygons>().template add_attribute<Type, false>(name);
		return PolygonAttributeHandle{
			static_cast<int32_t>(attr.index),
			type, false
		};
	}, [&type, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type ", get_attr_type_name(type));
		return INVALID_POLY_VATTR_HANDLE;
	});
	CATCH_ALL(INVALID_POLY_VATTR_HANDLE)
}

PolygonAttributeHandle polygon_request_face_attribute(ObjectHdl obj,
													  const char* name,
													  AttribDesc type) {
	TRY
	CHECK_NULLPTR(obj, "object handle", INVALID_POLY_FATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_POLY_FATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [name, type, &object](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		auto attr = object.template get_geometry<Polygons>().template add_attribute<Type, true>(name);
		return PolygonAttributeHandle{
			static_cast<int32_t>(attr.index),
			type, true
		};
	}, [&type, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type", get_attr_type_name(type));
		return INVALID_POLY_FATTR_HANDLE;
	});

	CATCH_ALL(INVALID_POLY_FATTR_HANDLE)
}

VertexHdl polygon_add_vertex(ObjectHdl obj, Vec3 point, Vec3 normal, Vec2 uv) {
	TRY
	CHECK_NULLPTR(obj, "object handle", VertexHdl{ INVALID_INDEX });
	PolyVHdl hdl = static_cast<Object*>(obj)->template get_geometry<Polygons>().add(
		util::pun<ei::Vec3>(point), util::pun<ei::Vec3>(normal),
		util::pun<ei::Vec2>(uv));
	if(!hdl.is_valid()) {
		logError("[", FUNCTION_NAME, "] Error adding vertex to polygon");
		return VertexHdl{ INVALID_INDEX };
	}
	return VertexHdl{ static_cast<IndexType>(hdl.idx()) };
	CATCH_ALL(VertexHdl{ INVALID_INDEX })
}

FaceHdl polygon_add_triangle(ObjectHdl obj, UVec3 vertices) {
	TRY
	CHECK_NULLPTR(obj, "object handle", FaceHdl{ INVALID_INDEX });

	PolyFHdl hdl = static_cast<Object*>(obj)->template get_geometry<Polygons>().add(
		PolyVHdl{ static_cast<int>(vertices.x) },
		PolyVHdl{ static_cast<int>(vertices.y) },
		PolyVHdl{ static_cast<int>(vertices.z) });
	if(!hdl.is_valid()) {
		logError("[", FUNCTION_NAME, "] Error adding triangle to polygon");
		return FaceHdl{ INVALID_INDEX };
	}
	return FaceHdl{ static_cast<IndexType>(hdl.idx()) };
	CATCH_ALL(FaceHdl{ INVALID_INDEX })
}

FaceHdl polygon_add_triangle_material(ObjectHdl obj, UVec3 vertices,
										 MatIdx idx) {
	TRY
	CHECK_NULLPTR(obj, "object handle", FaceHdl{ INVALID_INDEX });
	PolyFHdl hdl = static_cast<Object*>(obj)->template get_geometry<Polygons>().add(
		PolyVHdl{ static_cast<int>(vertices.x) },
		PolyVHdl{ static_cast<int>(vertices.y) },
		PolyVHdl{ static_cast<int>(vertices.z) },
		MaterialIndex{ idx });
	if(!hdl.is_valid()) {
		logError("[", FUNCTION_NAME, "] Error adding triangle to polygon");
		return FaceHdl{ INVALID_INDEX };
	}
	return FaceHdl{ static_cast<IndexType>(hdl.idx()) };
	CATCH_ALL(FaceHdl{ INVALID_INDEX })
}

FaceHdl polygon_add_quad(ObjectHdl obj, UVec4 vertices) {
	TRY
	CHECK_NULLPTR(obj, "object handle", FaceHdl{ INVALID_INDEX });
	PolyFHdl hdl = static_cast<Object*>(obj)->template get_geometry<Polygons>().add(
		PolyVHdl{ static_cast<int>(vertices.x) },
		PolyVHdl{ static_cast<int>(vertices.y) },
		PolyVHdl{ static_cast<int>(vertices.z) },
		PolyVHdl{ static_cast<int>(vertices.w) });
	if(!hdl.is_valid()) {
		logError("[", FUNCTION_NAME, "] Error adding triangle to polygon");
		return FaceHdl{ INVALID_INDEX };
	}
	return FaceHdl{ static_cast<IndexType>(hdl.idx()) };
	CATCH_ALL(FaceHdl{ INVALID_INDEX })
}

FaceHdl polygon_add_quad_material(ObjectHdl obj, UVec4 vertices,
									 MatIdx idx) {
	TRY
	CHECK_NULLPTR(obj, "object handle", FaceHdl{ INVALID_INDEX });
	PolyFHdl hdl = static_cast<Object*>(obj)->template get_geometry<Polygons>().add(
		PolyVHdl{ static_cast<int>(vertices.x) },
		PolyVHdl{ static_cast<int>(vertices.y) },
		PolyVHdl{ static_cast<int>(vertices.z) },
		PolyVHdl{ static_cast<int>(vertices.w) },
		MaterialIndex{ idx });
	if(!hdl.is_valid()) {
		logError("[", FUNCTION_NAME, "] Error adding triangle to polygon");
		return FaceHdl{ INVALID_INDEX };
	}
	return FaceHdl{ static_cast<IndexType>(hdl.idx()) };
	CATCH_ALL(FaceHdl{ INVALID_INDEX })
}

VertexHdl polygon_add_vertex_bulk(ObjectHdl obj, size_t count, FILE* points,
									 FILE* normals, FILE* uvs,
									 size_t* pointsRead, size_t* normalsRead,
									 size_t* uvsRead) {
	TRY
	CHECK_NULLPTR(obj, "object handle", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(points, "points stream descriptor", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(normals, "normals stream descriptor", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(uvs, "UV coordinates stream descriptor", VertexHdl{ INVALID_INDEX });
	Object& object = *static_cast<Object*>(obj);
	mufflon::util::FileReader pointReader{ points };
	mufflon::util::FileReader normalReader{ normals };
	mufflon::util::FileReader uvReader{ uvs };

	Polygons::VertexBulkReturn info = object.template get_geometry<Polygons>().add_bulk(
		count, pointReader, normalReader, uvReader);
	if(pointsRead != nullptr)
		*pointsRead = info.readPoints;
	if(pointsRead != nullptr)
		*normalsRead = info.readNormals;
	if(pointsRead != nullptr)
		*uvsRead = info.readUvs;
	return VertexHdl{ static_cast<IndexType>(info.handle.idx()) };
	CATCH_ALL(VertexHdl{ INVALID_INDEX })
}

VertexHdl polygon_add_vertex_bulk_no_normals(ObjectHdl obj, size_t count,
											 FILE* points, FILE* uvs,
											 size_t* pointsRead,
											 size_t* uvsRead) {
	TRY
	CHECK_NULLPTR(obj, "object handle", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(points, "points stream descriptor", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(uvs, "UV coordinates stream descriptor", VertexHdl{ INVALID_INDEX });
	Object& object = *static_cast<Object*>(obj);
	mufflon::util::FileReader pointReader{ points };
	mufflon::util::FileReader uvReader{ uvs };

	Polygons::VertexBulkReturn info = object.template get_geometry<Polygons>().add_bulk(
		count, pointReader, uvReader);
	if(pointsRead != nullptr)
		*pointsRead = info.readPoints;
	if(pointsRead != nullptr)
		*uvsRead = info.readUvs;
	return VertexHdl{ static_cast<IndexType>(info.handle.idx()) };
	CATCH_ALL(VertexHdl{ INVALID_INDEX })
}

VertexHdl polygon_add_vertex_bulk_aabb(ObjectHdl obj, size_t count, FILE* points,
										  FILE* normals, FILE* uvs, Vec3 min,
										  Vec3 max, size_t* pointsRead,
										  size_t* normalsRead, size_t* uvsRead) {
	TRY
	CHECK_NULLPTR(obj, "object handle", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(points, "points stream descriptor", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(normals, "normals stream descriptor", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(uvs, "UV coordinates stream descriptor", VertexHdl{ INVALID_INDEX });
	Object& object = *static_cast<Object*>(obj);
	mufflon::util::FileReader pointReader{ points };
	mufflon::util::FileReader normalReader{ normals };
	mufflon::util::FileReader uvReader{ uvs };

	ei::Box aabb{ util::pun<ei::Vec3>(min), util::pun<ei::Vec3>(max) };
	Polygons::VertexBulkReturn info = object.template get_geometry<Polygons>().add_bulk(
		count, pointReader, normalReader, uvReader, aabb);
	if(pointsRead != nullptr)
		*pointsRead = info.readPoints;
	if(pointsRead != nullptr)
		*normalsRead = info.readNormals;
	if(pointsRead != nullptr)
		*uvsRead = info.readUvs;
	return VertexHdl{ static_cast<IndexType>(info.handle.idx()) };
	CATCH_ALL(VertexHdl{ INVALID_INDEX })
}

VertexHdl polygon_add_vertex_bulk_aabb_no_normals(ObjectHdl obj, size_t count,
												  FILE* points, FILE* uvs,
												  Vec3 min, Vec3 max,
												  size_t* pointsRead,
												  size_t* uvsRead) {
	TRY
	CHECK_NULLPTR(obj, "object handle", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(points, "points stream descriptor", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(uvs, "UV coordinates stream descriptor", VertexHdl{ INVALID_INDEX });
	Object& object = *static_cast<Object*>(obj);
	mufflon::util::FileReader pointReader{ points };
	mufflon::util::FileReader uvReader{ uvs };

	ei::Box aabb{ util::pun<ei::Vec3>(min), util::pun<ei::Vec3>(max) };
	Polygons::VertexBulkReturn info = object.template get_geometry<Polygons>().add_bulk(
		count, pointReader, uvReader, aabb);
	if(pointsRead != nullptr)
		*pointsRead = info.readPoints;
	if(pointsRead != nullptr)
		*uvsRead = info.readUvs;
	return VertexHdl{ static_cast<IndexType>(info.handle.idx()) };
	CATCH_ALL(VertexHdl{ INVALID_INDEX })
}

Boolean polygon_set_vertex_attribute(ObjectHdl obj, const PolygonAttributeHandle* attr,
								  VertexHdl vertex, const void* value) {
	TRY
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(attr, "attribute handle", false);
	CHECK_NULLPTR(value, "attribute value", false);
	CHECK(!attr->face, "Face attribute in vertex function", false);
	CHECK_GEQ_ZERO(attr->index, "attribute index", false);
	CHECK_GEQ_ZERO(vertex, "vertex index", false);
	Object& object = *static_cast<Object*>(obj);
	if(vertex >= static_cast<int>(object.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 vertex, " >= ", object.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return false;
	}

	return switchAttributeType(attr->type, [&object, attr, vertex, value](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		OpenMeshAttributePool<false>::AttributeHandle hdl{ static_cast<std::size_t>(attr->index) };
		object.template get_geometry<Polygons>().acquire<Device::CPU, Type, false>(
			hdl)[vertex] = *static_cast<const Type*>(value);
		object.template get_geometry<Polygons>().mark_changed<false>(Device::CPU, hdl);
		return true;
	}, [attr, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(attr->type));
		return false;
	});
	CATCH_ALL(false)
}

Boolean polygon_set_vertex_normal(ObjectHdl obj, VertexHdl vertex, Vec3 normal) {
	TRY
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_GEQ_ZERO(vertex, "vertex index", false);
	Object& object = *static_cast<Object*>(obj);
	if(vertex >= static_cast<int>(object.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 vertex, " >= ", object.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return false;
	}

	auto hdl = object.get_geometry<Polygons>().get_normals_hdl();
	object.get_geometry<Polygons>().acquire<Device::CPU, OpenMesh::Vec3f, false>(hdl)[vertex] = util::pun<OpenMesh::Vec3f>(normal);
	object.get_geometry<Polygons>().mark_changed<false>(Device::CPU, hdl);
	return true;
	CATCH_ALL(false)
}

Boolean polygon_set_vertex_uv(ObjectHdl obj, VertexHdl vertex, Vec2 uv) {
	TRY
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_GEQ_ZERO(vertex, "vertex index", false);
	Object& object = *static_cast<Object*>(obj);
	if(vertex >= static_cast<int>(object.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 vertex, " >= ", object.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return false;
	}

	auto hdl = object.get_geometry<Polygons>().get_uvs_hdl();
	object.get_geometry<Polygons>().acquire<Device::CPU, OpenMesh::Vec2f, false>(hdl)[vertex] = util::pun<OpenMesh::Vec2f>(uv);
	object.get_geometry<Polygons>().mark_changed<false>(Device::CPU, hdl);
	return true;
	CATCH_ALL(false)
}

Boolean polygon_set_face_attribute(ObjectHdl obj, const PolygonAttributeHandle* attr,
								FaceHdl face, const void* value) {
	TRY
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(attr, "attribute handle", false);
	CHECK_NULLPTR(value, "attribute value", false);
	CHECK(attr->face, "Vertex attribute in face function", false);
	CHECK_GEQ_ZERO(face, "face index", false);
	CHECK_GEQ_ZERO(attr->index, "attribute index", false);
	Object& object = *static_cast<Object*>(obj);
	if(face >= static_cast<int>(object.template get_geometry<Polygons>().get_face_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 face, " >= ", object.template get_geometry<Polygons>().get_face_count(),
				 ")");
		return false;
	}

	return switchAttributeType(attr->type, [&object, attr, face, value](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		OpenMeshAttributePool<true>::AttributeHandle hdl{ static_cast<size_t>(attr->index) };
		object.template get_geometry<Polygons>().acquire<Device::CPU, Type, true>(
			hdl)[face] = *static_cast<const Type*>(value);
		object.template get_geometry<Polygons>().mark_changed<true>(Device::CPU, hdl);
		return true;
	}, [attr, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(attr->type));
		return false;
	});
	CATCH_ALL(false)
}

Boolean polygon_set_material_idx(ObjectHdl obj, FaceHdl face, MatIdx idx) {
	TRY
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_GEQ_ZERO(face, "face index", false);
	Object& object = *static_cast<Object*>(obj);
	if(face >= static_cast<int>(object.template get_geometry<Polygons>().get_face_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 face, " >= ", object.template get_geometry<Polygons>().get_face_count(),
				 ")");
		return false;
	}

	auto hdl = object.get_geometry<Polygons>().get_material_indices_hdl();
	object.get_geometry<Polygons>().acquire<Device::CPU, u16, true>(hdl)[face] = idx;
	object.get_geometry<Polygons>().mark_changed<true>(Device::CPU, hdl);
	return true;
	CATCH_ALL(false)
}

size_t polygon_set_vertex_attribute_bulk(ObjectHdl obj, const PolygonAttributeHandle* attr,
										 VertexHdl startVertex, size_t count,
										 FILE* stream) {
	TRY
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	CHECK_NULLPTR(attr, "attribute handle", INVALID_SIZE);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK(!attr->face, "Face attribute in vertex function", false);
	CHECK_GEQ_ZERO(startVertex, "start vertex index", INVALID_SIZE);
	CHECK_GEQ_ZERO(attr->index, "attribute index (OpenMesh)", INVALID_SIZE);
	Object& object = *static_cast<Object*>(obj);
	if(startVertex >= static_cast<int>(object.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 startVertex, " >= ", object.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return INVALID_SIZE;
	}
	util::FileReader attrStream{ stream };

	return switchAttributeType(attr->type, [&object, attr, startVertex, count, &attrStream](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		OpenMeshAttributePool<false>::AttributeHandle hdl{ static_cast<std::size_t>(attr->index) };
		return object.template get_geometry<Polygons>().add_bulk(hdl,
										 PolyVHdl{ static_cast<int>(startVertex) },
										 count, attrStream);
	}, [attr, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(attr->type));
		return INVALID_SIZE;
	});
	CATCH_ALL(INVALID_SIZE)
}

size_t polygon_set_face_attribute_bulk(ObjectHdl obj, const PolygonAttributeHandle* attr,
									   FaceHdl startFace, size_t count,
									   FILE* stream) {
	TRY
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	CHECK_NULLPTR(attr, "attribute handle", INVALID_SIZE);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK(attr->face, "Vertex attribute in face function", false);
	CHECK_GEQ_ZERO(startFace, "start face index", INVALID_SIZE);
	CHECK_GEQ_ZERO(attr->index, "attribute index (OpenMesh)", INVALID_SIZE);
	Object& object = *static_cast<Object*>(obj);
	if(startFace >= static_cast<int>(object.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 startFace, " >= ", object.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return INVALID_SIZE;
	}
	util::FileReader attrStream{ stream };

	return switchAttributeType(attr->type, [&object, attr, startFace, count, &attrStream](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		OpenMeshAttributePool<true>::AttributeHandle hdl{ static_cast<std::size_t>(attr->index) };
		return object.template get_geometry<Polygons>().add_bulk(hdl, PolyFHdl{ static_cast<int>(startFace) },
																 count, attrStream);
	}, [attr, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(attr->type));
		return INVALID_SIZE;
	});
	CATCH_ALL(INVALID_SIZE)
}

size_t polygon_set_material_idx_bulk(ObjectHdl obj, FaceHdl startFace, size_t count,
									 FILE* stream) {
	TRY
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK_GEQ_ZERO(startFace, "start face index", INVALID_SIZE);
	if(count == 0u)
		return 0u;
	Object& object = *static_cast<Object*>(obj);
	if(startFace >= static_cast<int>(object.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 startFace, " >= ", object.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return INVALID_SIZE;
	}
	util::FileReader matStream{ stream };

	OpenMeshAttributePool<true>::AttributeHandle hdl = object.template get_geometry<Polygons>().get_material_indices_hdl();
	return object.template get_geometry<Polygons>().add_bulk(hdl, PolyFHdl{ static_cast<int>(startFace) },
															 count, matStream);
	CATCH_ALL(INVALID_SIZE)
}

size_t polygon_get_vertex_count(ObjectHdl obj) {
	TRY
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.template get_geometry<Polygons>().get_vertex_count();
	CATCH_ALL(INVALID_SIZE)
}

size_t polygon_get_edge_count(ObjectHdl obj) {
	TRY
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.template get_geometry<Polygons>().get_edge_count();
	CATCH_ALL(INVALID_SIZE)
}

size_t polygon_get_face_count(ObjectHdl obj) {
	TRY
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.template get_geometry<Polygons>().get_face_count();
	CATCH_ALL(INVALID_SIZE)
}

size_t polygon_get_triangle_count(ObjectHdl obj) {
	TRY
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.template get_geometry<Polygons>().get_triangle_count();
	CATCH_ALL(INVALID_SIZE)
}

size_t polygon_get_quad_count(ObjectHdl obj) {
	TRY
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.template get_geometry<Polygons>().get_quad_count();
	CATCH_ALL(INVALID_SIZE)
}

Boolean polygon_get_bounding_box(ObjectHdl obj, Vec3* min, Vec3* max) {
	TRY
	CHECK_NULLPTR(obj, "object handle", false);
	const Object& object = *static_cast<const Object*>(obj);
	const ei::Box& aabb = object.template get_geometry<Polygons>().get_bounding_box();
	if(min != nullptr)
		*min = util::pun<Vec3>(aabb.min);
	if(max != nullptr)
		*max = util::pun<Vec3>(aabb.max);
	return true;
	CATCH_ALL(false)
}

Boolean spheres_reserve(ObjectHdl obj, size_t count) {
	TRY
	CHECK_NULLPTR(obj, "object handle", false);
	static_cast<Object*>(obj)->template get_geometry<Spheres>().reserve(count);
	return true;
	CATCH_ALL(false)
}

SphereAttributeHandle spheres_request_attribute(ObjectHdl obj, const char* name,
												AttribDesc type) {
	TRY
	CHECK_NULLPTR(obj, "object handle", INVALID_SPHERE_ATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_SPHERE_ATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [name, type, &object](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		return SphereAttributeHandle{
			static_cast<int32_t>(object.template get_geometry<Spheres>().add_attribute<Type>(name).index),
			type
		};
	}, [&type, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type", get_attr_type_name(type));
		return INVALID_SPHERE_ATTR_HANDLE;
	});
	CATCH_ALL(INVALID_SPHERE_ATTR_HANDLE)
}

SphereHdl spheres_add_sphere(ObjectHdl obj, Vec3 point, float radius) {
	TRY
	CHECK_NULLPTR(obj, "object handle", SphereHdl{ INVALID_INDEX });
	SphereVHdl hdl = static_cast<Object*>(obj)->template get_geometry<Spheres>().add(
		util::pun<ei::Vec3>(point), radius);
	return SphereHdl{ static_cast<IndexType>(hdl) };
	CATCH_ALL(SphereHdl{ INVALID_INDEX })
}

SphereHdl spheres_add_sphere_material(ObjectHdl obj, Vec3 point, float radius,
								MatIdx idx) {
	TRY
	CHECK_NULLPTR(obj, "object handle", SphereHdl{ INVALID_INDEX });
	SphereVHdl hdl = static_cast<Object*>(obj)->template get_geometry<Spheres>().add(
		util::pun<ei::Vec3>(point), radius, idx);
	return SphereHdl{ static_cast<IndexType>(hdl) };
	CATCH_ALL(SphereHdl{ INVALID_INDEX })
}

SphereHdl spheres_add_sphere_bulk(ObjectHdl obj, size_t count,
									 FILE* stream, size_t* readSpheres) {
	TRY
	CHECK_NULLPTR(obj, "object handle", SphereHdl{ INVALID_INDEX } );
	CHECK_NULLPTR(stream, "sphere stream descriptor", SphereHdl{ INVALID_INDEX });
	Object& object = *static_cast<Object*>(obj);
	mufflon::util::FileReader sphereReader{ stream };

	Spheres::BulkReturn info = object.template get_geometry<Spheres>().add_bulk(count, sphereReader);
	if(readSpheres != nullptr)
		*readSpheres = info.readSpheres;
	return SphereHdl{ static_cast<IndexType>(info.handle) };
	CATCH_ALL(SphereHdl{ INVALID_INDEX })
}

SphereHdl spheres_add_sphere_bulk_aabb(ObjectHdl obj, size_t count,
										  FILE* stream, Vec3 min, Vec3 max,
										  size_t* readSpheres) {
	TRY
	CHECK_NULLPTR(obj, "object handle", SphereHdl{ INVALID_INDEX });
	CHECK_NULLPTR(stream, "sphere stream descriptor", SphereHdl{ INVALID_INDEX });
	Object& object = *static_cast<Object*>(obj);
	mufflon::util::FileReader sphereReader{ stream };

	ei::Box aabb{ util::pun<ei::Vec3>(min), util::pun<ei::Vec3>(max) };
	Spheres::BulkReturn info = object.template get_geometry<Spheres>().add_bulk(count, sphereReader,
														aabb);
	if(readSpheres != nullptr)
		*readSpheres = info.readSpheres;
	return SphereHdl{ static_cast<IndexType>(info.handle) };
	CATCH_ALL(SphereHdl{ INVALID_INDEX })
}

Boolean spheres_set_attribute(ObjectHdl obj, const SphereAttributeHandle* attr,
						   SphereHdl sphere, const void* value) {
	TRY
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(attr, "attribute handle", false);
	CHECK_NULLPTR(value, "attribute value", false);
	CHECK_GEQ_ZERO(sphere, "sphere index", false);
	CHECK_GEQ_ZERO(attr->index, "attribute index", false);
	Object& object = *static_cast<Object*>(obj);
	if(sphere >= static_cast<int>(object.template get_geometry<Spheres>().get_sphere_count())) {
		logError("[", FUNCTION_NAME, "] Sphere index out of bounds (",
				 sphere, " >= ", object.template get_geometry<Spheres>().get_sphere_count(),
				 ")");
		return false;
	}

	return switchAttributeType(attr->type, [&object, attr, sphere, value](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		AttributePool::AttributeHandle hdl{ static_cast<std::size_t>(attr->index) };
		object.template get_geometry<Spheres>().acquire<Device::CPU, Type>(hdl)[sphere] = *static_cast<const Type*>(value);
		object.template get_geometry<Spheres>().mark_changed(Device::CPU, hdl);
		return true;
	}, [attr, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(attr->type));
		return false;
	});
	CATCH_ALL(false)
}

Boolean spheres_set_material_idx(ObjectHdl obj, SphereHdl sphere, MatIdx idx) {
	TRY
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_GEQ_ZERO(sphere, "sphere index", false);
	Object& object = *static_cast<Object*>(obj);
	if(sphere >= static_cast<int>(object.template get_geometry<Spheres>().get_sphere_count())) {
		logError("[", FUNCTION_NAME, "] Sphere index out of bounds (",
				 sphere, " >= ", object.template get_geometry<Spheres>().get_sphere_count(),
				 ")");
		return false;
	}

	AttributePool::AttributeHandle hdl = object.template get_geometry<Spheres>().get_material_indices_hdl();
	object.template get_geometry<Spheres>().acquire<Device::CPU, u16>(hdl)[sphere] = idx;
	object.template get_geometry<Spheres>().mark_changed(Device::CPU, hdl);
	return true;
	CATCH_ALL(false)
}

size_t spheres_set_attribute_bulk(ObjectHdl obj, const SphereAttributeHandle* attr,
								  SphereHdl startSphere, size_t count,
								  FILE* stream) {
	TRY
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	CHECK_NULLPTR(attr, "attribute handle", INVALID_SIZE);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK_GEQ_ZERO(startSphere, "start sphere index", INVALID_SIZE);
	CHECK_GEQ_ZERO(attr->index, "attribute index", INVALID_SIZE);
	Object& object = *static_cast<Object*>(obj);
	if(startSphere >= static_cast<int>(object.template get_geometry<Spheres>().get_sphere_count())) {
		logError("[", FUNCTION_NAME, "] Sphere index out of bounds (",
				 startSphere, " >= ", object.template get_geometry<Spheres>().get_sphere_count(),
				 ")");
		return INVALID_SIZE;
	}
	util::FileReader attrStream{ stream };

	return switchAttributeType(attr->type, [&object, attr, startSphere, count, &attrStream](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		AttributePool::AttributeHandle hdl{ static_cast<std::size_t>(attr->index) };
		return object.template get_geometry<Spheres>().add_bulk(hdl,
																SphereVHdl{ static_cast<size_t>(startSphere) },
																count, attrStream);
	}, [attr, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(attr->type));
		return INVALID_SIZE;
	});
	CATCH_ALL(INVALID_SIZE)
}

size_t spheres_set_material_idx_bulk(ObjectHdl obj, SphereHdl startSphere, size_t count,
									 FILE* stream) {
	TRY
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK_GEQ_ZERO(startSphere, "start sphere index", INVALID_SIZE);
	Object& object = *static_cast<Object*>(obj);
	if(startSphere >= static_cast<int>(object.template get_geometry<Spheres>().get_sphere_count())) {
		logError("[", FUNCTION_NAME, "] Sphere index out of bounds (",
				 startSphere, " >= ", object.template get_geometry<Spheres>().get_sphere_count(),
				 ")");
		return INVALID_SIZE;
	}
	util::FileReader matStream{ stream };

	AttributePool::AttributeHandle hdl = object.template get_geometry<Spheres>().get_material_indices_hdl();
	return object.template get_geometry<Spheres>().add_bulk(hdl,
															SphereVHdl{ static_cast<size_t>(startSphere) },
															count, matStream);
	return INVALID_SIZE;
	CATCH_ALL(INVALID_SIZE)
}

size_t spheres_get_sphere_count(ObjectHdl obj) {
	TRY
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.template get_geometry<Spheres>().get_sphere_count();
	CATCH_ALL(INVALID_SIZE)
}

Boolean spheres_get_bounding_box(ObjectHdl obj, Vec3* min, Vec3* max) {
	TRY
	CHECK_NULLPTR(obj, "object handle", false);
	const Object& object = *static_cast<const Object*>(obj);
	const ei::Box& aabb = object.get_bounding_box<Spheres>();
	if(min != nullptr)
		*min = util::pun<Vec3>(aabb.min);
	if(max != nullptr)
		*max = util::pun<Vec3>(aabb.max);
	return true;
	CATCH_ALL(false)
}

Boolean instance_set_transformation_matrix(InstanceHdl inst, const Mat3x4* mat) {
	TRY
	CHECK_NULLPTR(inst, "instance handle", false);
	CHECK_NULLPTR(mat, "transformation matrix", false);
	Instance& instance = *static_cast<InstanceHandle>(inst);
	instance.set_transformation_matrix(util::pun<ei::Mat3x4>(*mat));
	return true;
	CATCH_ALL(false)
}

Boolean instance_get_transformation_matrix(InstanceHdl inst, Mat3x4* mat) {
	TRY
	CHECK_NULLPTR(inst, "instance handle", false);
	const Instance& instance = *static_cast<ConstInstanceHandle>(inst);
	if(mat != nullptr)
		*mat = util::pun<Mat3x4>(instance.get_transformation_matrix());
	return true;
	CATCH_ALL(false)
}

Boolean instance_get_bounding_box(InstanceHdl inst, Vec3* min, Vec3* max) {
	TRY
	CHECK_NULLPTR(inst, "instance handle", false);
	const Instance& instance = *static_cast<ConstInstanceHandle>(inst);
	const ei::Box& aabb = instance.get_bounding_box();
	if(min != nullptr)
		*min = util::pun<Vec3>(aabb.min);
	if(max != nullptr)
		*max = util::pun<Vec3>(aabb.max);
	return true;
	CATCH_ALL(false)
}

void world_clear_all() {
	TRY
	WorldContainer::clear_instance();
	CATCH_ALL(;)
}

ObjectHdl world_create_object(const char* name, ::ObjectFlags flags) {
	TRY
	CHECK_NULLPTR(name, "object name", nullptr);
	return static_cast<ObjectHdl>(s_world.create_object(name,
			util::pun<scene::ObjectFlags>(flags)));
	CATCH_ALL(nullptr)
}

ObjectHdl world_get_object(const char* name) {
	TRY
	CHECK_NULLPTR(name, "object name", nullptr);
	return static_cast<ObjectHdl>(s_world.get_object(name));
	CATCH_ALL(nullptr)
}

CORE_API const char* CDECL world_get_object_name(ObjectHdl obj) {
	TRY
	CHECK_NULLPTR(obj, "object handle", nullptr);
	return &reinterpret_cast<mufflon::scene::ConstObjectHandle>(obj)->get_name()[0];
	CATCH_ALL(nullptr)
}

InstanceHdl world_create_instance(ObjectHdl obj) {
	TRY
	CHECK_NULLPTR(obj, "object handle", nullptr);
	ObjectHandle hdl = static_cast<Object*>(obj);
	return static_cast<InstanceHdl>(s_world.create_instance(hdl));
	CATCH_ALL(nullptr)
}

ScenarioHdl world_create_scenario(const char* name) {
	TRY
	CHECK_NULLPTR(name, "scenario name", nullptr);
	ScenarioHandle hdl = s_world.create_scenario(name);
	return static_cast<ScenarioHdl>(hdl);
	CATCH_ALL(nullptr)
}

ScenarioHdl world_find_scenario(const char* name) {
	TRY
	CHECK_NULLPTR(name, "scenario name", nullptr);
	std::string_view nameView{ name };
	ScenarioHandle hdl = s_world.get_scenario(nameView);
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Could not find scenario '",
				 nameView, "'");
		return nullptr;
	}
	return static_cast<ScenarioHdl>(hdl);
	CATCH_ALL(nullptr)
}

uint32_t world_get_scenario_count() {
	TRY
	return static_cast<uint32_t>(s_world.get_scenario_count());
	CATCH_ALL(0u)
}

ScenarioHdl world_get_scenario_by_index(uint32_t index) {
	TRY
	const uint32_t MAX_INDEX = static_cast<uint32_t>(s_world.get_scenario_count());
	if(index >= MAX_INDEX) {
		logError("[", FUNCTION_NAME, "] Scenario index '", index, "' out of bounds (",
				 MAX_INDEX, ')');
		return nullptr;
	}
	return s_world.get_scenario(index);
	CATCH_ALL(nullptr)
}

ConstScenarioHdl world_get_current_scenario() {
	TRY
	return static_cast<ConstScenarioHdl>(s_world.get_current_scenario());
	CATCH_ALL(nullptr)
}

namespace {
std::unique_ptr<materials::IMaterial> convert_material(const char* name, const MaterialParams* mat) {
	CHECK_NULLPTR(name, "material name", nullptr);
	CHECK_NULLPTR(mat, "material parameters", nullptr);

	std::unique_ptr<materials::IMaterial> newMaterial;
	switch(mat->innerType) {
		case MATERIAL_LAMBERT: {
			auto tex = mat->inner.lambert.albedo;
			newMaterial = std::make_unique<materials::Lambert>(static_cast<TextureHandle>(tex));
		}	break;
		case MATERIAL_TORRANCE:
			// TODO
			logWarning("[", FUNCTION_NAME, "] Material type 'torrance' not supported yet");
			return nullptr;
		case MATERIAL_WALTER:
			logWarning("[", FUNCTION_NAME, "] Material type 'walter' not supported yet");
			return nullptr;
		case MATERIAL_EMISSIVE: {
			auto tex = mat->inner.emissive.radiance;
			newMaterial = std::make_unique<materials::Emissive>(static_cast<TextureHandle>(tex),
								util::pun<Spectrum>(mat->inner.emissive.scale));
		}	break;
		case MATERIAL_ORENNAYAR:
			logWarning("[", FUNCTION_NAME, "] Material type 'orennayar' not supported yet");
			return nullptr;
		case MATERIAL_BLEND: {
			auto a = convert_material("LayerA", mat->inner.blend.a.mat);
			auto b = convert_material("LayerB", mat->inner.blend.b.mat);
			newMaterial = std::make_unique<materials::Blend>(
				move(a), mat->inner.blend.a.factor,
				move(b), mat->inner.blend.b.factor);
		}	break;
		case MATERIAL_FRESNEL:
			logWarning("[", FUNCTION_NAME, "] Material type 'fresnel' not supported yet");
			return nullptr;
		default:
			logWarning("[", FUNCTION_NAME, "] Unknown material type");
	}

	// Set common properties and add to scene
	newMaterial->set_name(name);
	newMaterial->set_outer_medium( s_world.add_medium(
		{util::pun<ei::Vec2>(mat->outerMedium.refractionIndex),
			util::pun<Spectrum>(mat->outerMedium.absorption)}) );
	newMaterial->set_inner_medium( s_world.add_medium(newMaterial->compute_medium()) );

	return move(newMaterial);
}
} // namespace ::

MaterialHdl world_add_material(const char* name, const MaterialParams* mat) {
	TRY
	MaterialHandle hdl = s_world.add_material(convert_material(name, mat));

	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Error creating material '",
				 name, "'");
		return nullptr;
	}

	return static_cast<MaterialHdl>(hdl);
	CATCH_ALL(nullptr)
}


CameraHdl world_add_pinhole_camera(const char* name, Vec3 position, Vec3 dir,
								   Vec3 up, float near, float far, float vFov) {
	TRY
	CHECK_NULLPTR(name, "camera name", nullptr);
	CameraHandle hdl = s_world.add_camera(name,
		std::make_unique<cameras::Pinhole>(
			util::pun<ei::Vec3>(position), util::pun<ei::Vec3>(dir),
			util::pun<ei::Vec3>(up), vFov, near, far
	));
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Error creating pinhole camera");
		return nullptr;
	}
	return static_cast<CameraHdl>(hdl);
	CATCH_ALL(nullptr)
}

CameraHdl world_add_focus_camera(const char* name, Vec3 position, Vec3 dir,
								 Vec3 up, float near, float far,
								 float focalLength, float focusDistance,
								 float lensRad, float chipHeight) {
	TRY
	CHECK_NULLPTR(name, "camera name", nullptr);
	CameraHandle hdl = s_world.add_camera(name,
		std::make_unique<cameras::Focus>(
			util::pun<ei::Vec3>(position), util::pun<ei::Vec3>(dir),
			util::pun<ei::Vec3>(up), focalLength, focusDistance,
			lensRad, chipHeight, near, far
		));
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Error creating focus camera");
		return nullptr;
	}
	return static_cast<CameraHdl>(hdl);
	CATCH_ALL(nullptr)
}

Boolean world_remove_camera(CameraHdl hdl) {
	TRY
	CHECK_NULLPTR(hdl, "camera handle", false);
	s_world.remove_camera(static_cast<CameraHandle>(hdl));
	return true;
	CATCH_ALL(false)
}

LightHdl world_add_light(const char* name, LightType type) {
	TRY
	CHECK_NULLPTR(name, "pointlight name", (LightHdl{7, 0}));
	std::optional<u32> hdl;
	switch(type) {
		case LIGHT_POINT: {
			hdl = s_world.add_light(name, lights::PointLight{});
		} break;
		case LIGHT_SPOT: {
			hdl = s_world.add_light(name, lights::SpotLight{});
		} break;
		case LIGHT_DIRECTIONAL: {
			hdl = s_world.add_light(name, lights::DirectionalLight{});
		} break;
		case LIGHT_ENVMAP: {
			hdl = s_world.add_light(name, TextureHandle{});
		} break;
	}
	if(!hdl.has_value()) {
		logError("[", FUNCTION_NAME, "] Error adding a light");
		return LightHdl{7, 0};
	}
	return LightHdl{u32(type), hdl.value()};
	CATCH_ALL((LightHdl{7, 0}))
}

Boolean world_remove_light(LightHdl hdl) {
	TRY
	s_world.remove_light(hdl.index, lights::LightType(hdl.type));
	return true;
	CATCH_ALL(false)
}

Boolean world_find_light(const char* name, LightHdl* hdl) {
	TRY
	CHECK_NULLPTR(name, "light name", false);
	std::optional<std::pair<u32, lights::LightType>> light = s_world.find_light(name);
	if(!light.has_value())
		return false;
	if(hdl != nullptr)
		*hdl = LightHdl{ static_cast<u32>(light.value().second), light.value().first };
	return true;
	CATCH_ALL(false);
}

size_t world_get_camera_count() {
	TRY
	return s_world.get_camera_count();
	CATCH_ALL(0u)
}

CameraHdl world_get_camera(const char* name) {
	TRY
	CHECK_NULLPTR(name, "camera name", nullptr);
	return static_cast<CameraHdl>(s_world.get_camera(name));
	CATCH_ALL(nullptr)
}

CORE_API CameraHdl CDECL world_get_camera_by_index(size_t index) {
	TRY
	return static_cast<CameraHdl>(s_world.get_camera(index));
	CATCH_ALL(0u)
}

size_t world_get_point_light_count() {
	TRY
	return s_world.get_point_light_count();
	CATCH_ALL(0u)
}

size_t world_get_spot_light_count() {
	TRY
	return s_world.get_spot_light_count();
	CATCH_ALL(0u)
}

size_t world_get_dir_light_count() {
	TRY
	return s_world.get_dir_light_count();
	CATCH_ALL(0u)
}

size_t world_get_env_light_count() {
	TRY
	return s_world.get_env_light_count();
	CATCH_ALL(0u)
}

CORE_API LightHdl CDECL world_get_light_handle(size_t index, LightType type) {
	return LightHdl{u32(type), u32(index)};
}

CORE_API const char* CDECL world_get_light_name(LightHdl hdl) {
	constexpr lights::LightType TYPES[] = {
		lights::LightType::POINT_LIGHT,
		lights::LightType::SPOT_LIGHT,
		lights::LightType::DIRECTIONAL_LIGHT,
		lights::LightType::ENVMAP_LIGHT
	};
	return s_world.get_light_name(hdl.index, TYPES[hdl.type]).data();
}

SceneHdl world_load_scenario(ScenarioHdl scenario) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	SceneHandle hdl = s_world.load_scene(static_cast<ScenarioHandle>(scenario));
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Failed to load scenario");
		return nullptr;
	}
	ei::IVec2 res = static_cast<ConstScenarioHandle>(scenario)->get_resolution();
	if(s_currentRenderer != nullptr)
		s_currentRenderer->load_scene(hdl, res);
	s_imageOutput = std::make_unique<renderer::OutputHandler>(res.x, res.y, s_outputTargets);
	return static_cast<SceneHdl>(hdl);
	CATCH_ALL(nullptr)
}

SceneHdl world_reload_current_scenario() {
	TRY
	SceneHandle hdl = s_world.reload_scene();
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Failed to reload scenario");
		return nullptr;
	}
	ei::IVec2 res = s_world.get_current_scenario()->get_resolution();
	if(s_currentRenderer != nullptr)
		s_currentRenderer->reset();
	s_imageOutput = std::make_unique<renderer::OutputHandler>(res.x, res.y, s_outputTargets);
	return static_cast<SceneHdl>(hdl);
	CATCH_ALL(nullptr)
}

SceneHdl world_get_current_scene() {
	TRY
	return static_cast<SceneHdl>(s_world.get_current_scene());
	CATCH_ALL(nullptr)
}

Boolean world_is_sane(const char** msg) {
	TRY
	switch(s_world.is_sane_world()) {
		case WorldContainer::Sanity::SANE: *msg = "";  return true;
		case WorldContainer::Sanity::NO_CAMERA: *msg = "No camera"; return false;
		case WorldContainer::Sanity::NO_INSTANCES: *msg = "No instances"; return false;
		case WorldContainer::Sanity::NO_OBJECTS: *msg = "No objects"; return false;
		case WorldContainer::Sanity::NO_LIGHTS: *msg = "No lights or emitters"; return false;
	}
	return false;
	CATCH_ALL(false)
}

TextureHdl world_get_texture(const char* path) {
	TRY
	CHECK_NULLPTR(path, "texture path", false);
	auto hdl = s_world.find_texture(path);
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Could not find texture ",
				 path);
		return nullptr;
	}
	return static_cast<TextureHdl>(hdl);
	CATCH_ALL(nullptr)
}

TextureHdl world_add_texture(const char* path, TextureSampling sampling) {
	TRY
	CHECK_NULLPTR(path, "texture path", nullptr);

	// Check if the texture is already loaded
	auto hdl = s_world.find_texture(path);
	if(hdl != nullptr) {
		s_world.ref_texture(hdl);
		return static_cast<TextureHdl>(hdl);
	}

	// Use the plugins to load the texture
	fs::path filePath(path);
	TextureData texData{};
	for(auto& plugin : s_plugins) {
		if(plugin.is_loaded()) {
			if(plugin.can_load_format(filePath.extension().string())) {
				if(plugin.load(filePath.string(), &texData))
					break;
			}
		}
	}
	if(texData.data == nullptr) {
		logError("[", FUNCTION_NAME, "] No plugin could load texture '",
				 filePath.string(), "'");
		return nullptr;
	}
	// The texture will take ownership of the pointer
	hdl = s_world.add_texture(path, texData.width, texData.height,
												 texData.layers, static_cast<textures::Format>(texData.format),
												 static_cast<textures::SamplingMode>(sampling),
												 texData.sRgb, std::unique_ptr<u8[]>(texData.data));
	return static_cast<TextureHdl>(hdl);
	CATCH_ALL(nullptr)
}

TextureHdl world_add_texture_value(const float* value, int num, TextureSampling sampling) {
	TRY
	mAssert(num >= 1 && num <= 4);
	// Create an artifical name for the value texture (for compatibilty with file-textures)
	std::string name = std::to_string(value[0]);
	for(int i = 1; i < num; ++i)
		name += " " + std::to_string(value[i]);

	// Check if the texture is already loaded
	auto hdl = s_world.find_texture(name);
	if(hdl != nullptr) {
		s_world.ref_texture(hdl);
		return static_cast<TextureHdl>(hdl);
	}

	textures::Format format;
	switch(num) {
		case 1: format = textures::Format::R32F; break;
		case 2: format = textures::Format::RG32F; break;
		case 3: format = textures::Format::RGBA32F; break;
		case 4: format = textures::Format::RGBA32F; break;
		default:
			logError("[", FUNCTION_NAME, "] Invalid number of channels (", num, ')');
			return nullptr;
	}

	// Create new
	ei::Vec4 paddedVal{0.0f};
	memcpy(&paddedVal, value, num * sizeof(float));
	std::unique_ptr<u8[]> data = std::make_unique<u8[]>(textures::PIXEL_SIZE(format));
	memcpy(data.get(), &paddedVal, textures::PIXEL_SIZE(format));
	hdl = s_world.add_texture(name, 1, 1, 1, format,
							  static_cast<textures::SamplingMode>(sampling),
							  false, move(data));
	return static_cast<TextureHdl>(hdl);
	CATCH_ALL(nullptr)
}

CameraType world_get_camera_type(ConstCameraHdl cam) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", CameraType::CAM_COUNT);
	cameras::CameraModel type = static_cast<const cameras::Camera*>(cam)->get_model();
	switch(type) {
		case cameras::CameraModel::PINHOLE: return CameraType::CAM_PINHOLE;
		case cameras::CameraModel::FOCUS: return CameraType::CAM_FOCUS;
		default: return CameraType::CAM_COUNT;
	}
	CATCH_ALL(CameraType::CAM_COUNT)
}

const char* world_get_camera_name(ConstCameraHdl cam) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", nullptr);
	std::string_view name = static_cast<const cameras::Camera*>(cam)->get_name();
	return &name[0];
	CATCH_ALL(nullptr)
}

Boolean world_get_camera_position(ConstCameraHdl cam, Vec3* pos) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	if(pos != nullptr)
		*pos = util::pun<Vec3>(static_cast<const cameras::Camera*>(cam)->get_position());
	return true;
	CATCH_ALL(false)
}

Boolean world_get_camera_direction(ConstCameraHdl cam, Vec3* dir) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	if(dir != nullptr)
		*dir = util::pun<Vec3>(static_cast<const cameras::Camera*>(cam)->get_view_dir());
	return true;
	CATCH_ALL(false)
}

Boolean world_get_camera_up(ConstCameraHdl cam, Vec3* up) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	if(up != nullptr)
		*up = util::pun<Vec3>(static_cast<const cameras::Camera*>(cam)->get_up_dir());
	return true;
	CATCH_ALL(false)
}

Boolean world_get_camera_near(ConstCameraHdl cam, float* near) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	if(near != nullptr)
		*near = static_cast<const cameras::Camera*>(cam)->get_near();
	return true;
	CATCH_ALL(false)
}

Boolean world_get_camera_far(ConstCameraHdl cam, float* far) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	if(far != nullptr)
		*far = static_cast<const cameras::Camera*>(cam)->get_far();
	return true;
	CATCH_ALL(false)
}

Boolean world_set_camera_position(CameraHdl cam, Vec3 pos) {
	TRY
		CHECK_NULLPTR(cam, "camera handle", false);
	auto& camera = *static_cast<cameras::Camera*>(cam);
	scene::Point newPos = camera.get_position() - util::pun<scene::Point>(pos);
	camera.move(newPos.x, newPos.y, newPos.z);
	s_world.mark_camera_dirty(static_cast<CameraHandle>(cam));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_camera_direction(CameraHdl cam, Vec3 dir) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	auto& camera = *static_cast<cameras::Camera*>(cam);
	// TODO: compute proper rotation
	s_world.mark_camera_dirty(static_cast<CameraHandle>(cam));
	return false;
	CATCH_ALL(false)
}

Boolean world_set_camera_up(CameraHdl cam, Vec3 up) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	auto& camera = *static_cast<cameras::Camera*>(cam);
	// TODO: ???
	s_world.mark_camera_dirty(static_cast<CameraHandle>(cam));
	return false;
	CATCH_ALL(false)
}

Boolean world_set_camera_near(CameraHdl cam, float near) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	CHECK(near > 0.f, "near-plane", false);
	static_cast<cameras::Camera*>(cam)->set_near(near);
	s_world.mark_camera_dirty(static_cast<CameraHandle>(cam));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_camera_far(CameraHdl cam, float far) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	CHECK(far > 0.f, "far-plane", false);
	static_cast<cameras::Camera*>(cam)->set_far(far);
	s_world.mark_camera_dirty(static_cast<CameraHandle>(cam));
	return true;
	CATCH_ALL(false)
}

Boolean world_get_pinhole_camera_fov(ConstCameraHdl cam, float* vFov) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	if(vFov != nullptr)
		*vFov = static_cast<const cameras::Pinhole*>(cam)->get_vertical_fov();
	return true;
	CATCH_ALL(false)
}

Boolean world_set_pinhole_camera_fov(CameraHdl cam, float vFov) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	CHECK(vFov > 0.f && vFov < 180.f, "vertical field-of-view", false);
	static_cast<cameras::Pinhole*>(cam)->set_vertical_fov(vFov);
	s_world.mark_camera_dirty(static_cast<CameraHandle>(cam));
	return true;
	CATCH_ALL(false)
}

Boolean world_get_focus_camera_focal_length(ConstCameraHdl cam, float* focalLength) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	if(focalLength != nullptr)
		*focalLength = static_cast<const cameras::Focus*>(cam)->get_focal_length();
	return true;
	CATCH_ALL(false)
}

Boolean world_get_focus_camera_focus_distance(ConstCameraHdl cam, float* focusDistance) {
	TRY
		CHECK_NULLPTR(cam, "camera handle", false);
	if(focusDistance != nullptr)
		*focusDistance = static_cast<const cameras::Focus*>(cam)->get_focus_distance();
	return true;
	CATCH_ALL(false)
}

Boolean world_get_focus_camera_sensor_height(ConstCameraHdl cam, float* sensorHeight) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	if(sensorHeight != nullptr)
		*sensorHeight = static_cast<const cameras::Focus*>(cam)->get_sensor_height();
	return true;
	CATCH_ALL(false)
}

Boolean world_get_focus_camera_aperture(ConstCameraHdl cam, float* aperture) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	const auto& camera = *static_cast<const cameras::Focus*>(cam);
	if(aperture != nullptr)
		*aperture =  camera.get_focal_length() / (2.f * camera.get_lens_radius());
	return true;
	CATCH_ALL(false)
}

Boolean world_set_focus_camera_focal_length(CameraHdl cam, float focalLength) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	CHECK(focalLength > 0.f, "focalLength", false);
	static_cast<cameras::Focus*>(cam)->set_focal_length(focalLength);
	s_world.mark_camera_dirty(static_cast<CameraHandle>(cam));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_focus_camera_focus_distance(CameraHdl cam, float focusDistance) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	CHECK(focusDistance > 0.f, "focus distance", false);
	static_cast<cameras::Focus*>(cam)->set_focus_distance(focusDistance);
	s_world.mark_camera_dirty(static_cast<CameraHandle>(cam));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_focus_camera_sensor_height(CameraHdl cam, float sensorHeight) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	CHECK(sensorHeight > 0.f, "sensor height", false);
	static_cast<cameras::Focus*>(cam)->set_sensor_height(sensorHeight);
	s_world.mark_camera_dirty(static_cast<CameraHandle>(cam));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_focus_camera_aperture(CameraHdl cam, float aperture) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	CHECK(aperture > 0.f, "aperture", false);
	auto& camera = *static_cast<cameras::Focus*>(cam);
	camera.set_lens_radius(camera.get_focal_length() / aperture);
	s_world.mark_camera_dirty(static_cast<CameraHandle>(cam));
	return true;
	CATCH_ALL(false)
}


const char* scenario_get_name(ScenarioHdl scenario) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	// This relies on the fact that the string_view in scenario points to
	// an std::string object, which is null terminated
	return &static_cast<const Scenario*>(scenario)->get_name()[0u];
	CATCH_ALL(nullptr)
}

size_t scenario_get_global_lod_level(ScenarioHdl scenario) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", Scenario::NO_CUSTOM_LOD);
	return static_cast<const Scenario*>(scenario)->get_global_lod_level();
	CATCH_ALL(Scenario::NO_CUSTOM_LOD)
}

Boolean scenario_set_global_lod_level(ScenarioHdl scenario, LodLevel level) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	static_cast<Scenario*>(scenario)->set_global_lod_level(level);
	return true;
	CATCH_ALL(false)
}

Boolean scenario_get_resolution(ScenarioHdl scenario, uint32_t* width, uint32_t* height) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	ei::IVec2 res = static_cast<const Scenario*>(scenario)->get_resolution();
	if(width != nullptr)
		*width = res.x;
	if(height != nullptr)
		*height = res.y;
	return true;
	CATCH_ALL(false)
}

Boolean scenario_set_resolution(ScenarioHdl scenario, uint32_t width, uint32_t height) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	static_cast<Scenario*>(scenario)->set_resolution(ei::IVec2{ width, height });
	return true;
	CATCH_ALL(false)
}

CameraHdl scenario_get_camera(ScenarioHdl scenario) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	return static_cast<CameraHdl>(static_cast<const Scenario*>(scenario)->get_camera());
	CATCH_ALL(nullptr)
}

Boolean scenario_set_camera(ScenarioHdl scenario, CameraHdl cam) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	static_cast<Scenario*>(scenario)->set_camera(static_cast<CameraHandle>(cam));
	if(scenario == world_get_current_scenario())
		s_world.get_current_scene()->set_camera(static_cast<CameraHandle>(cam));
	return true;
	CATCH_ALL(false)
}

Boolean scenario_is_object_masked(ScenarioHdl scenario, ObjectHdl obj) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(obj, "object handle", false);
	return static_cast<const Scenario*>(scenario)->is_masked(static_cast<const Object*>(obj));
	CATCH_ALL(false)
}

Boolean scenario_mask_object(ScenarioHdl scenario, ObjectHdl obj) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(obj, "object handle", false);
	static_cast<Scenario*>(scenario)->mask_object(static_cast<const Object*>(obj));
	return true;
	CATCH_ALL(false)
}

LodLevel scenario_get_object_lod(ScenarioHdl scenario, ObjectHdl obj) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", Scenario::NO_CUSTOM_LOD);
	CHECK_NULLPTR(obj, "object handle", Scenario::NO_CUSTOM_LOD);
	return static_cast<const Scenario*>(scenario)->get_custom_lod(static_cast<const Object*>(obj));
	CATCH_ALL(Scenario::NO_CUSTOM_LOD)
}

Boolean scenario_set_object_lod(ScenarioHdl scenario, ObjectHdl obj,
							 LodLevel level) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(obj, "object handle", false);
	static_cast<Scenario*>(scenario)->set_custom_lod(static_cast<const Object*>(obj),
													 level);
	return true;
	CATCH_ALL(false)
}

IndexType scenario_get_point_light_count(ScenarioHdl scenario) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_INDEX);
	return static_cast<IndexType>(static_cast<const Scenario*>(scenario)->get_point_light_names().size());
	CATCH_ALL(INVALID_INDEX)
}

IndexType scenario_get_spot_light_count(ScenarioHdl scenario) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_INDEX);
	return static_cast<IndexType>(static_cast<const Scenario*>(scenario)->get_spot_light_names().size());
	CATCH_ALL(INVALID_INDEX)
}

IndexType scenario_get_dir_light_count(ScenarioHdl scenario) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_INDEX);
	return static_cast<IndexType>(static_cast<const Scenario*>(scenario)->get_dir_light_names().size());
	CATCH_ALL(INVALID_INDEX)
}

Boolean scenario_has_envmap_light(ScenarioHdl scenario) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	return !static_cast<IndexType>(static_cast<const Scenario*>(scenario)->get_envmap_light_name().empty());
	CATCH_ALL(INVALID_INDEX)
}

const char* scenario_get_point_light_name(ScenarioHdl scenario, size_t index) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	const Scenario& scen = *static_cast<const Scenario*>(scenario);
	if(index >= scen.get_point_light_names().size()) {
		logError("[", FUNCTION_NAME, "] Light index out of bounds (",
				 index, " >= ", scen.get_point_light_names().size(), ")");
		return nullptr;
	}
	std::string_view name = scen.get_point_light_names()[index];
	return &name[0];
	CATCH_ALL(nullptr)
}

const char* scenario_get_spot_light_name(ScenarioHdl scenario, size_t index) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	const Scenario& scen = *static_cast<const Scenario*>(scenario);
	if(index >= scen.get_spot_light_names().size()) {
		logError("[", FUNCTION_NAME, "] Light index out of bounds (",
				 index, " >= ", scen.get_spot_light_names().size(), ")");
		return nullptr;
	}
	std::string_view name = scen.get_spot_light_names()[index];
	return &name[0];
	CATCH_ALL(nullptr)
}

const char* scenario_get_dir_light_name(ScenarioHdl scenario, size_t index) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	const Scenario& scen = *static_cast<const Scenario*>(scenario);
	if(index >= scen.get_dir_light_names().size()) {
		logError("[", FUNCTION_NAME, "] Light index out of bounds (",
				 index, " >= ", scen.get_dir_light_names().size(), ")");
		return nullptr;
	}
	std::string_view name = scen.get_dir_light_names()[index];
	return &name[0];
	CATCH_ALL(nullptr)
}

const char* scenario_get_envmap_light_name(ScenarioHdl scenario) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	const Scenario& scen = *static_cast<const Scenario*>(scenario);
	std::string_view name = scen.get_envmap_light_name();
	return &name[0];
	CATCH_ALL(nullptr)
}

Boolean scenario_add_light(ScenarioHdl scenario, const char* name) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(name, "light name", false);
	Scenario& scen = *static_cast<Scenario*>(scenario);
	std::string_view nameView = name;
	std::optional<std::pair<u32, lights::LightType>> hdl = s_world.find_light(nameView);
	if(!hdl.has_value()) {
		logError("[", FUNCTION_NAME, "] Light source '", nameView, "' does not exist");
		return false;
	}
	std::string_view resolvedName = s_world.get_light_name(hdl.value().first, hdl.value().second);

	switch(hdl.value().second) {
		case lights::LightType::POINT_LIGHT:
			scen.add_point_light(resolvedName);
			return true;
		case lights::LightType::SPOT_LIGHT:
			scen.add_spot_light(resolvedName);
			return true;
		case lights::LightType::DIRECTIONAL_LIGHT:
			scen.add_dir_light(resolvedName);
			return true;
		case lights::LightType::ENVMAP_LIGHT:
		{
			if(!scen.get_envmap_light_name().empty())
				logWarning("[", FUNCTION_NAME, "] The scenario already has an envmap light; overwriting '",
						   scen.get_envmap_light_name(), "' with '", resolvedName, "'");
			scen.set_envmap_light(resolvedName);
			return true;
		default:
			logError("[", FUNCTION_NAME, "] Unknown or invalid light type");
			return false;
		}
	}
	CATCH_ALL(false)
}

Boolean scenario_remove_light(ScenarioHdl scenario, const char* name) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(name, "light name", false);
	Scenario& scen = *static_cast<Scenario*>(scenario);
	std::string_view nameView = name;
	scen.remove_point_light(nameView);
	scen.remove_spot_light(nameView);
	scen.remove_dir_light(nameView);
	if(nameView == scen.get_envmap_light_name())
		scen.remove_envmap_light();
	return true;
	CATCH_ALL(false)
}

MatIdx scenario_declare_material_slot(ScenarioHdl scenario,
									  const char* name, std::size_t nameLength) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_MATERIAL);
	CHECK_NULLPTR(name, "material name", INVALID_MATERIAL);
	std::string_view nameView(name, std::min<std::size_t>(nameLength, std::strlen(name)));
	return static_cast<Scenario*>(scenario)->declare_material_slot(nameView);
	CATCH_ALL(INVALID_MATERIAL)
}

MatIdx scenario_get_material_slot(ScenarioHdl scenario,
								  const char* name, std::size_t nameLength) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_MATERIAL);
	CHECK_NULLPTR(name, "material name", INVALID_MATERIAL);
	std::string_view nameView(name, std::min<std::size_t>(nameLength, std::strlen(name)));
	return static_cast<const Scenario*>(scenario)->get_material_slot_index(nameView);
	CATCH_ALL(INVALID_MATERIAL)
}

MaterialHdl scenario_get_assigned_material(ScenarioHdl scenario,
										   MatIdx index) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	CHECK((index < INVALID_MATERIAL), "Invalid material index", nullptr);
	return static_cast<MaterialHdl>(static_cast<const Scenario*>(scenario)->get_assigned_material(index));
	CATCH_ALL(nullptr)
}

Boolean scenario_assign_material(ScenarioHdl scenario, MatIdx index,
							  MaterialHdl handle) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(handle, "material handle", false);
	CHECK((index < INVALID_MATERIAL), "Invalid material index", false);
	static_cast<Scenario*>(scenario)->assign_material(index, static_cast<MaterialHandle>(handle));
	return true;
	CATCH_ALL(false)
}

Boolean scenario_is_sane(ConstScenarioHdl scenario, const char** msg) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	switch(s_world.is_sane_scenario(static_cast<ConstScenarioHandle>(scenario))) {
		case WorldContainer::Sanity::SANE: *msg = "";  return true;
		case WorldContainer::Sanity::NO_CAMERA: *msg = "No camera"; return false;
		case WorldContainer::Sanity::NO_INSTANCES: *msg = "No instances"; return false;
		case WorldContainer::Sanity::NO_OBJECTS: *msg = "No objects"; return false;
		case WorldContainer::Sanity::NO_LIGHTS: *msg = "No lights or emitters"; return false;
	}
	return false;
	CATCH_ALL(false)
}

Boolean scene_get_bounding_box(SceneHdl scene, Vec3* min, Vec3* max) {
	TRY
	CHECK_NULLPTR(scene, "scene handle", false);
	const Scene& scen = *static_cast<const Scene*>(scene);
	if(min != nullptr)
		*min = util::pun<Vec3>(scen.get_bounding_box().min);
	if(max != nullptr)
		*max = util::pun<Vec3>(scen.get_bounding_box().max);
	return true;
	CATCH_ALL(false)
}

ConstCameraHdl scene_get_camera(SceneHdl scene) {
	TRY
	CHECK_NULLPTR(scene, "scene handle", nullptr);
	return static_cast<ConstCameraHdl>(static_cast<const Scene*>(scene)->get_camera());
	CATCH_ALL(nullptr)
}

Boolean scene_move_active_camera(float x, float y, float z) {
	TRY
	if(s_world.get_current_scene() == nullptr) {
		logError("[", FUNCTION_NAME, "] No scene loaded yet");
		return false;
	}
	s_world.get_current_scenario()->get_camera()->move(x, y, z);
	s_world.mark_camera_dirty(s_world.get_current_scenario()->get_camera());
	return true;
	CATCH_ALL(false)
}

Boolean scene_rotate_active_camera(float x, float y, float z) {
	TRY
		if(s_world.get_current_scene() == nullptr) {
			logError("[", FUNCTION_NAME, "] No scene loaded yet");
			return false;
		}
	s_world.get_current_scenario()->get_camera()->rotate_up_down(x);
	s_world.get_current_scenario()->get_camera()->rotate_left_right(y);
	s_world.get_current_scenario()->get_camera()->roll(z);
	s_world.mark_camera_dirty(s_world.get_current_scenario()->get_camera());
	return true;
	CATCH_ALL(false)
}

Boolean scene_is_sane() {
	TRY
	ConstSceneHandle sceneHdl = s_world.get_current_scene();
	if(sceneHdl != nullptr)
		return sceneHdl->is_sane();
	return false;
	CATCH_ALL(false)
}

Boolean world_get_point_light_position(ConstLightHdl hdl, Vec3* pos) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_POINT, "light type must be point", false);
	const lights::PointLight* light = s_world.get_point_light(hdl.index);
	CHECK_NULLPTR(light, "point light handle", false);
	if(pos != nullptr)
		*pos = util::pun<Vec3>(light->position);
	return true;
	CATCH_ALL(false)
}

Boolean world_get_point_light_intensity(ConstLightHdl hdl, Vec3* intensity) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_POINT, "light type must be point", false);
	const lights::PointLight* light = s_world.get_point_light(hdl.index);
	CHECK_NULLPTR(light, "point light handle", false);
	if(intensity != nullptr)
		*intensity = util::pun<Vec3>(light->intensity);
	return true;
	CATCH_ALL(false)
}

Boolean world_set_point_light_position(LightHdl hdl, Vec3 pos) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_POINT, "light type must be point", false);
	lights::PointLight* light = s_world.get_point_light(hdl.index);
	CHECK_NULLPTR(light, "point light handle", false);
	light->position = util::pun<ei::Vec3>(pos);
	s_world.mark_light_dirty(hdl.index, static_cast<lights::LightType>(hdl.type));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_point_light_intensity(LightHdl hdl, Vec3 intensity) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_POINT, "light type must be point", false);
	lights::PointLight* light = s_world.get_point_light(hdl.index);
	CHECK_NULLPTR(light, "point light handle", false);
	light->intensity = util::pun<ei::Vec3>(intensity);
	s_world.mark_light_dirty(hdl.index, static_cast<lights::LightType>(hdl.type));
	return true;
	CATCH_ALL(false)
}

Boolean world_get_spot_light_position(ConstLightHdl hdl, Vec3* pos) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	const lights::SpotLight* light = s_world.get_spot_light(hdl.index);
	CHECK_NULLPTR(light, "spot light handle", false);
	if(pos != nullptr)
		*pos = util::pun<Vec3>(light->position);
	return true;
	CATCH_ALL(false)
}

Boolean world_get_spot_light_intensity(ConstLightHdl hdl, Vec3* intensity) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	const lights::SpotLight* light = s_world.get_spot_light(hdl.index);
	CHECK_NULLPTR(light, "spot light handle", false);
	if(intensity != nullptr)
		*intensity = util::pun<Vec3>(light->intensity);
	return true;
	CATCH_ALL(false)
}

Boolean world_get_spot_light_direction(ConstLightHdl hdl, Vec3* direction) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	const lights::SpotLight* light = s_world.get_spot_light(hdl.index);
	CHECK_NULLPTR(light, "spot light handle", false);
	if(direction != nullptr)
		*direction = util::pun<Vec3>(light->direction);
	return true;
	CATCH_ALL(false)
}

Boolean world_get_spot_light_angle(ConstLightHdl hdl, float* angle) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	const lights::SpotLight* light = s_world.get_spot_light(hdl.index);
	CHECK_NULLPTR(light, "spot light handle", false);
	if(angle != nullptr)
		*angle = std::acos(__half2float(light->cosThetaMax));
	return true;
	CATCH_ALL(false)
}

Boolean world_get_spot_light_falloff(ConstLightHdl hdl, float* falloff) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	const lights::SpotLight* light = s_world.get_spot_light(hdl.index);
	CHECK_NULLPTR(light, "spot light handle", false);
	if(falloff != nullptr)
		*falloff = std::acos(__half2float(light->cosFalloffStart));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_spot_light_position(LightHdl hdl, Vec3 pos) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	lights::SpotLight* light = s_world.get_spot_light(hdl.index);
	CHECK_NULLPTR(light, "spot light handle", false);
	light->position = util::pun<ei::Vec3>(pos);
	s_world.mark_light_dirty(hdl.index, static_cast<lights::LightType>(hdl.type));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_spot_light_intensity(LightHdl hdl, Vec3 intensity) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	lights::SpotLight* light = s_world.get_spot_light(hdl.index);
	CHECK_NULLPTR(light, "spot light handle", false);
	light->intensity = util::pun<ei::Vec3>(intensity);
	s_world.mark_light_dirty(hdl.index, static_cast<lights::LightType>(hdl.type));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_spot_light_direction(LightHdl hdl, Vec3 direction) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	lights::SpotLight* light = s_world.get_spot_light(hdl.index);
	CHECK_NULLPTR(light, "spot light handle", false);
	ei::Vec3 actualDirection = ei::normalize(util::pun<ei::Vec3>(direction));
	if(!ei::approx(ei::len(actualDirection), 1.0f)) {
		logError("[", FUNCTION_NAME, "] Spotlight direction cannot be a null vector");
		return false;
	}
	light->direction = util::pun<ei::Vec3>(actualDirection);
	s_world.mark_light_dirty(hdl.index, static_cast<lights::LightType>(hdl.type));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_spot_light_angle(LightHdl hdl, float angle) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	lights::SpotLight* light = s_world.get_spot_light(hdl.index);
	CHECK_NULLPTR(light, "spot light handle", false);

	float actualAngle = std::fmod(angle, 2.f * ei::PI);
	if(actualAngle < 0.f)
		actualAngle += 2.f*ei::PI;
	if(actualAngle > ei::PI / 2.f) {
		logWarning("[", FUNCTION_NAME, "] Spotlight angle will be clamped between 0-180 degrees");
		actualAngle = ei::PI / 2.f;
	}
	light->cosThetaMax = __float2half(std::cos(actualAngle));
	s_world.mark_light_dirty(hdl.index, static_cast<lights::LightType>(hdl.type));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_spot_light_falloff(LightHdl hdl, float falloff) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	lights::SpotLight* light = s_world.get_spot_light(hdl.index);
	CHECK_NULLPTR(light, "spot light handle", false);
	// Clamp it to the opening angle!
	float actualFalloff = std::fmod(falloff, 2.f * ei::PI);
	if(actualFalloff < 0.f)
		actualFalloff += 2.f*ei::PI;
	const float cosFalloff = std::cos(actualFalloff);
	const __half compressedCosFalloff = __float2half(cosFalloff);
	if(__half2float(light->cosThetaMax) > __half2float(compressedCosFalloff)) {
		logWarning("[", FUNCTION_NAME, "] Spotlight falloff angle cannot be larger than"
				   " its opening angle");
		light->cosFalloffStart = light->cosThetaMax;
	} else {
		light->cosFalloffStart = compressedCosFalloff;
	}
	s_world.mark_light_dirty(hdl.index, static_cast<lights::LightType>(hdl.type));
	return true;
	CATCH_ALL(false)
}

Boolean world_get_dir_light_direction(ConstLightHdl hdl, Vec3* direction) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_DIRECTIONAL, "light type must be directional", false);
	const lights::DirectionalLight* light = s_world.get_dir_light(hdl.index);
	CHECK_NULLPTR(light, "directional light handle", false);
	if(direction != nullptr)
		*direction = util::pun<Vec3>(light->direction);
	return true;
	CATCH_ALL(false)
}

Boolean world_get_dir_light_irradiance(ConstLightHdl hdl, Vec3* irradiance) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_DIRECTIONAL, "light type must be directional", false);
	const lights::DirectionalLight* light = s_world.get_dir_light(hdl.index);
	CHECK_NULLPTR(light, "directional light handle", false);
	if(irradiance != nullptr)
		*irradiance = util::pun<Vec3>(light->irradiance);
	return true;
	CATCH_ALL(false)
}

Boolean world_set_dir_light_direction(LightHdl hdl, Vec3 direction) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_DIRECTIONAL, "light type must be directional", false);
	lights::DirectionalLight* light = s_world.get_dir_light(hdl.index);
	CHECK_NULLPTR(light, "directional light handle", false);
	ei::Vec3 actualDirection = ei::normalize(util::pun<ei::Vec3>(direction));
	if(!ei::approx(ei::len(actualDirection), 1.0f)) {
		logError("[", FUNCTION_NAME, "] Directional light direction cannot be a null vector");
		return false;
	}
	light->direction = util::pun<ei::Vec3>(actualDirection);
	s_world.mark_light_dirty(hdl.index, static_cast<lights::LightType>(hdl.type));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_dir_light_irradiance(LightHdl hdl, Vec3 irradiance) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_DIRECTIONAL, "light type must be directional", false);
	lights::DirectionalLight* light = s_world.get_dir_light(hdl.index);
	CHECK_NULLPTR(light, "directional light handle", false);
	light->irradiance = util::pun<ei::Vec3>(irradiance);
	s_world.mark_light_dirty(hdl.index, static_cast<lights::LightType>(hdl.type));
	return true;
	CATCH_ALL(false)
}

const char* world_get_env_light_map(ConstLightHdl hdl) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be envmap", false);
	ConstTextureHandle envmap = s_world.get_env_light(hdl.index)->get_envmap();
	CHECK_NULLPTR(envmap, "environment-mapped light handle", false);
	auto nameOpt = s_world.get_texture_name(envmap);
	if(!nameOpt.has_value()) {
		logError("[", FUNCTION_NAME, "] Could not find envmap path!");
		return nullptr;
	}

	std::string_view path = nameOpt.value();
	return &path[0];
	CATCH_ALL(nullptr)
}

Boolean world_set_env_light_map(LightHdl hdl, TextureHdl tex) {
	TRY
	CHECK_NULLPTR(tex, "texture handle", false);
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be envmap", false);
	s_world.replace_envlight_texture(hdl.index, reinterpret_cast<TextureHandle>(tex));
	s_world.mark_light_dirty(hdl.index, static_cast<lights::LightType>(hdl.type));
	return true;
	CATCH_ALL(false)
}

CORE_API Boolean CDECL world_set_env_light_scale(LightHdl hdl, Vec3 color) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be envmap", false);
	s_world.get_env_light(hdl.index)->set_scale(util::pun<Spectrum>(color));
	s_world.mark_light_dirty(hdl.index, static_cast<lights::LightType>(hdl.type));
	return true;
	CATCH_ALL(false)
}

Boolean render_enable_renderer(RendererType type) {
	TRY
	switch(type) {
		case RendererType::RENDERER_CPU_PT: {
			s_currentRenderer = std::make_unique<renderer::CpuPathTracer>();
		}	break;
		case RendererType::RENDERER_GPU_PT: {
			s_currentRenderer = std::make_unique<renderer::GpuPathTracer>();
		}	break;
		default: {
			logError("[", FUNCTION_NAME, "] Unknown renderer type");
			return false;
		}
	}
	if(s_world.get_current_scenario() != nullptr)
		s_currentRenderer->load_scene(s_world.get_current_scene(),
			s_world.get_current_scenario()->get_resolution());
	return true;
	CATCH_ALL(false)
}

Boolean render_iterate() {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] No renderer is currently set");
		return false;
	}
	if(s_imageOutput == nullptr) {
		logError("[", FUNCTION_NAME, "] No rendertarget is currently set");
		return false;
	}
	if(!s_currentRenderer->has_scene()) {
		logError("[", FUNCTION_NAME, "] Scene not yet set for renderer");
		return false;
	}
	s_currentRenderer->iterate(*s_imageOutput);
	return true;
	CATCH_ALL(false)
}

Boolean render_reset() {
	TRY
	if(s_currentRenderer != nullptr)
		s_currentRenderer->reset();
	return true;
	CATCH_ALL(false)
}

Boolean render_save_screenshot(const char* filename) {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] No renderer is currently set");
		return false;
	}

	// TODO: this is just for debugging! This should be done by an image library
	auto dumpPfm = [](std::string fileName, const uint8_t* data) {
		std::ofstream file(fileName, std::ofstream::binary | std::ofstream::out);
		if(file.bad()) {
			logError("[", FUNCTION_NAME, "] Failed to open screenshot file '",
					 fileName, "'");
			return;
		}
		file.write("PF\n", 3);
		ei::IVec2 res = s_imageOutput->get_resolution();
		auto sizes = std::to_string(res.x) + " " + std::to_string(res.y);
		file.write(sizes.c_str(), sizes.length());
		file.write("\n-1.000000\n", 11);

		const auto pixels = reinterpret_cast<const char *>(data);
		for(int y = 0; y < res.y; ++y) {
			for(int x = 0; x < res.x; ++x) {
				file.write(&pixels[(y * res.x + x) * 4u * sizeof(float)], 3u * sizeof(float));
			}
		}

	};
	const std::string name(filename);
	for(u32 target : renderer::OutputValue::iterator) {
		if(s_outputTargets.is_set(target)) {
			textures::CpuTexture data = s_imageOutput->get_data(renderer::OutputValue{ target }, textures::Format::RGBA32F, false);
			dumpPfm(name + "_" + std::to_string(target) + ".pfm", data.data());
		}
		if(s_outputTargets.is_set(target << 8u)) {
			textures::CpuTexture data = s_imageOutput->get_data(renderer::OutputValue{ target << 8u }, textures::Format::RGBA32F, false);
			dumpPfm(name + "_" + std::to_string(target) + "_var.pfm", data.data());
		}
	}
	logInfo("[", FUNCTION_NAME, "] Saved screenshot '", filename, "'");

	return true;
	CATCH_ALL(false)
}

Boolean render_enable_render_target(RenderTarget target, Boolean variance) {
	TRY
	switch(target) {
		case RenderTarget::TARGET_RADIANCE:
			s_outputTargets.set(renderer::OutputValue::RADIANCE << (variance ? 8u : 0u));
			break;
		case RenderTarget::TARGET_POSITION:
			s_outputTargets.set(renderer::OutputValue::POSITION << (variance ? 8u : 0u));
			break;
		case RenderTarget::TARGET_ALBEDO:
			s_outputTargets.set(renderer::OutputValue::ALBEDO << (variance ? 8u : 0u));
			break;
		case RenderTarget::TARGET_NORMAL:
			s_outputTargets.set(renderer::OutputValue::NORMAL << (variance ? 8u : 0u));
			break;
		case RenderTarget::TARGET_LIGHTNESS:
			s_outputTargets.set(renderer::OutputValue::LIGHTNESS << (variance ? 8u : 0u));
			break;
		default:
			logError("[", FUNCTION_NAME, "] Unknown render target");
			return false;
	}
	if(s_imageOutput != nullptr)
		s_imageOutput->set_targets(s_outputTargets);
	return true;
	CATCH_ALL(false)
}

Boolean render_disable_render_target(RenderTarget target, Boolean variance) {
	TRY
	switch(target) {
		case RenderTarget::TARGET_RADIANCE:
			s_outputTargets.clear(renderer::OutputValue::RADIANCE << (variance ? 8u : 0u));
			break;
		case RenderTarget::TARGET_POSITION:
			s_outputTargets.clear(renderer::OutputValue::POSITION << (variance ? 8u : 0u));
			break;
		case RenderTarget::TARGET_ALBEDO:
			s_outputTargets.clear(renderer::OutputValue::ALBEDO << (variance ? 8u : 0u));
			break;
		case RenderTarget::TARGET_NORMAL:
			s_outputTargets.clear(renderer::OutputValue::NORMAL << (variance ? 8u : 0u));
			break;
		case RenderTarget::TARGET_LIGHTNESS:
			s_outputTargets.clear(renderer::OutputValue::LIGHTNESS << (variance ? 8u : 0u));
			break;
		default:
			logError("[", FUNCTION_NAME, "] Unknown render target");
			return false;
	}
	if(s_imageOutput != nullptr)
		s_imageOutput->set_targets(s_outputTargets);
	return true;
	CATCH_ALL(false)
}

Boolean render_enable_variance_render_targets() {
	TRY
	for(u32 target : renderer::OutputValue::iterator)
		s_outputTargets.set(target << 8u);
	if(s_imageOutput != nullptr)
		s_imageOutput->set_targets(s_outputTargets);
	return true;
	CATCH_ALL(false)
}

Boolean render_enable_non_variance_render_targets() {
	TRY
	for(u32 target : renderer::OutputValue::iterator)
		s_outputTargets.set(target);
	if(s_imageOutput != nullptr)
		s_imageOutput->set_targets(s_outputTargets);
	return true;
	CATCH_ALL(false)
}

Boolean render_enable_all_render_targets() {
	TRY
	return render_enable_variance_render_targets()
		&& render_enable_non_variance_render_targets();
	CATCH_ALL(false)
}

Boolean render_disable_variance_render_targets() {
	TRY
	for(u32 target : renderer::OutputValue::iterator)
		s_outputTargets.clear(target << 8u);
	if(s_imageOutput != nullptr)
		s_imageOutput->set_targets(s_outputTargets);
	return true;
	CATCH_ALL(false)
}

Boolean render_disable_non_variance_render_targets() {
	TRY
	for(u32 target : renderer::OutputValue::iterator)
		s_outputTargets.clear(target);
	if(s_imageOutput != nullptr)
		s_imageOutput->set_targets(s_outputTargets);
	return true;
	CATCH_ALL(false)
}

Boolean render_disable_all_render_targets() {
	TRY
	return render_disable_variance_render_targets()
		&& render_disable_non_variance_render_targets();
	CATCH_ALL(false)
}

uint32_t renderer_get_num_parameters() {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return 0u;
	}
	return s_currentRenderer->get_parameters().get_num_parameters();
	CATCH_ALL(0u)
}

const char* renderer_get_parameter_desc(uint32_t idx, ParameterType* type) {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return nullptr;
	}
	renderer::ParamDesc rendererDesc = s_currentRenderer->get_parameters().get_param_desc(idx);

	if(type != nullptr)
		*type = static_cast<ParameterType>(rendererDesc.type);
	return rendererDesc.name;
	CATCH_ALL(nullptr)
}

Boolean renderer_set_parameter_int(const char* name, int32_t value) {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	s_currentRenderer->get_parameters().set_param(name, value);
	return true;
	CATCH_ALL(false)
}

Boolean renderer_get_parameter_int(const char* name, int32_t* value) {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	*value = s_currentRenderer->get_parameters().get_param_int(name);
	return true;
	CATCH_ALL(false)
}

Boolean renderer_set_parameter_float(const char* name, float value) {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	s_currentRenderer->get_parameters().set_param(name, value);
	return true;
	CATCH_ALL(false)
}

Boolean renderer_get_parameter_float(const char* name, float* value) {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	*value = s_currentRenderer->get_parameters().get_param_float(name);
	return true;
	CATCH_ALL(false)
}

Boolean renderer_set_parameter_bool(const char* name, Boolean value) {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	s_currentRenderer->get_parameters().set_param(name, bool(value));
	return true;
	CATCH_ALL(false)
}

Boolean renderer_get_parameter_bool(const char* name, Boolean* value) {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	*value = s_currentRenderer->get_parameters().get_param_bool(name);
	return true;
	CATCH_ALL(false)
}

void profiling_enable() {
	TRY
	Profiler::instance().set_enabled(true);
	CATCH_ALL(;)
}

void profiling_disable() {
	TRY
	Profiler::instance().set_enabled(false);
	CATCH_ALL(;)
}

Boolean profiling_set_level(ProfilingLevel level) {
	TRY
	switch(level) {
		case ProfilingLevel::PROFILING_OFF:
			Profiler::instance().set_enabled(false);
			return true;
		case ProfilingLevel::PROFILING_LOW:
			Profiler::instance().set_profile_level(ProfileLevel::LOW);
			return true;
		case ProfilingLevel::PROFILING_HIGH:
			Profiler::instance().set_profile_level(ProfileLevel::HIGH);
			return true;
		case ProfilingLevel::PROFILING_ALL:
			Profiler::instance().set_profile_level(ProfileLevel::ALL);
			return true;
		default:
			logError("[", FUNCTION_NAME, "] invalid profiling level");
			return false;
	}
	CATCH_ALL(false)
}

Boolean profiling_save_current_state(const char* path) {
	TRY
	CHECK_NULLPTR(path, "file path", false);
	Profiler::instance().save_current_state(path);
	return true;
	CATCH_ALL(false)
}

Boolean profiling_save_snapshots(const char* path) {
	TRY
	CHECK_NULLPTR(path, "file path", false);
	Profiler::instance().save_snapshots(path);
	return true;
	CATCH_ALL(false)
}

Boolean profiling_save_total_and_snapshots(const char* path) {
	TRY
	CHECK_NULLPTR(path, "file path", false);
	Profiler::instance().save_total_and_snapshots(path);
	return true;
	CATCH_ALL(false)
}

const char* profiling_get_current_state() {
	TRY
	static std::string str = Profiler::instance().save_current_state();
	return str.c_str();
	CATCH_ALL(nullptr)
}

const char* profiling_get_snapshots() {
	TRY
	static std::string str = Profiler::instance().save_snapshots();
	return str.c_str();
	CATCH_ALL(nullptr)
}

const char* profiling_get_total_and_snapshots() {
	TRY
	static std::string str = Profiler::instance().save_total_and_snapshots();
	return str.c_str();
	CATCH_ALL(nullptr)
}

void profiling_reset() {
	TRY
	Profiler::instance().reset_all();
	CATCH_ALL(;)
}

size_t profiling_get_total_cpu_memory() {
	TRY
	return CpuProfileState::get_total_memory();
	CATCH_ALL(0u)
}

size_t profiling_get_free_cpu_memory() {
	TRY
	return CpuProfileState::get_free_memory();
	CATCH_ALL(0u)
}

size_t profiling_get_used_cpu_memory() {
	TRY
	return CpuProfileState::get_used_memory();
	CATCH_ALL(0u)
}

size_t profiling_get_total_gpu_memory() {
	TRY
	return GpuProfileState::get_total_memory();
	CATCH_ALL(0u)
}

size_t profiling_get_free_gpu_memory() {
	TRY
	return GpuProfileState::get_free_memory();
	CATCH_ALL(0u)
}

size_t profiling_get_used_gpu_memory() {
	TRY
	return GpuProfileState::get_used_memory();
	CATCH_ALL(0u)
}

Boolean mufflon_initialize(void(*logCallback)(const char*, int)) {
	TRY
	// Only once per process do we register/unregister the message handler
	static bool initialized = false;
	s_logCallback = logCallback;
	if(!initialized) {
		registerMessageHandler(delegateLog);
		disableStdHandler();

		// Load plugins from the DLLs directory
		s_plugins.clear();
		fs::path dllPath;
		// First obtain the module handle (platform specific), then use that to
		// get the module's path
#ifdef _WIN32
		HMODULE moduleHandle = nullptr;
		if(::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
							 | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
							 reinterpret_cast<LPCTSTR>(&mufflon_initialize),
							 &moduleHandle)) {
			WCHAR buffer[MAX_PATH] = { 0 };
			DWORD length = ::GetModuleFileNameW(moduleHandle, buffer, MAX_PATH);
			if(length == 0)
				logError("[", FUNCTION_NAME, "] Failed to obtain module path; cannot load plugins");
			else
				dllPath = std::wstring(buffer, length);
		} else {
			logError("[", FUNCTION_NAME, "] Failed to obtain module handle; cannot load plugins");
		}
#else // _WIN32
		Dl_info info;
		if(::dladdr(reinterpret_cast<void*>(mufflon_initialize), &info) == 0)
			logError("[", FUNCTION_NAME, "] Failed to obtain module path; cannot load plugins");
		else
			dllPath = info.dli_fname;
#endif // _WIN32
		// If we managed to get the module path, check for plugins there
		if(!dllPath.empty()) {
			// We assume that the plugins are in a subdirectory called 'plugins' to
			// avoid loading excessively many DLLs
			for(const auto& dir : fs::directory_iterator(dllPath.parent_path() / "plugins")) {
				fs::path path = dir.path();
				if(!fs::is_directory(path) && path.extension() == ".dll") {
					TextureLoaderPlugin plugin{ path };
					// If we succeeded in loading (and thus have the necessary functions),
					// add it as a usable plugin
					if(plugin.is_loaded()) {
						logInfo("[", FUNCTION_NAME, "] Loaded texture plugin '",
								plugin.get_path().string(), "'");
						plugin.set_logger(s_logCallback);
						s_plugins.push_back(std::move(plugin));
					}
				}
			}
		}

		if(!gladLoadGL()) {
			logError("[", FUNCTION_NAME, "] gladLoadGL failed");
			return false;
		}

		// Set the CUDA device to initialize the context
		int count = 0;
		cuda::check_error(cudaGetDeviceCount(&count));
		if(count > 0) {
			// We select the device with the highest compute capability
			int devIndex = -1;
			int major = -1;
			int minor = -1;

			cudaDeviceProp deviceProp;
			for (int c = 0; c < count; ++c) {
				cudaGetDeviceProperties(&deviceProp, c);
				if(deviceProp.unifiedAddressing) {
					if(deviceProp.major > major ||
						((deviceProp.major == major) && (deviceProp.minor > minor))) {
						major = deviceProp.major;
						minor = deviceProp.minor;
						devIndex = c;
					}
				}
			}
			if(devIndex < 0) {
				logWarning("[", FUNCTION_NAME, "] Found CUDA device(s), but none supports unified addressing; "
						 "continuing without CUDA");
			} else {
				cuda::check_error(cudaSetDevice(devIndex));
				s_cudaDevIndex = devIndex;
				logInfo("[", FUNCTION_NAME, "] Found ", count, " CUDA-capable "
						"devices; initializing device ", devIndex, " (", deviceProp.name, ")");
			}
		} else {
			logInfo("[", FUNCTION_NAME, "] No CUDA device found; continuing without CUDA");
		}

		initialized = true;
	}
	return initialized;
	CATCH_ALL(false)
}

int32_t mufflon_get_cuda_device_index() {
	TRY
	return s_cudaDevIndex;
	CATCH_ALL(-1)
}

Boolean mufflon_is_cuda_available() {
	TRY
	int count = 0;
	cuda::check_error(cudaGetDeviceCount(&count));
	if(count > 0) {
		cudaDeviceProp deviceProp;
		for(int c = 0; c < count; ++c) {
			cudaGetDeviceProperties(&deviceProp, c);
			if(deviceProp.unifiedAddressing) 
				return true;
		}
	}
	return false;
	CATCH_ALL(false)
}

void mufflon_destroy() {
	TRY
	WorldContainer::clear_instance();
	s_imageOutput.reset();
	s_currentRenderer.reset();
	CATCH_ALL(;)
}

/*const char* get_teststring() {
	static const char* test = u8"müfflon ηρα Φ∞∡∧";
	return test;
}*/
