#include "interface.h"
#include "plugin/texture_plugin.hpp"
#include "util/log.hpp"
#include "util/byte_io.hpp"
#include "util/punning.hpp"
#include "util/degrad.hpp"
#include "ei/vector.hpp"
#include "profiler/cpu_profiler.hpp"
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
#include "loader/export/interface.h"
#include <cuda_runtime.h>
#include <type_traits>
#include <mutex>
#include <fstream>
#include <vector>
// TODO: remove this (leftover from Felix' prototype)
#include <glad/glad.h>

#ifdef _WIN32
#include <minwindef.h>
#else // _WIN32
#include <dlfcn.h>
#endif // _WIN32

// Undefine unnecessary windows macros
#undef near
#undef far

using namespace mufflon;
using namespace mufflon::scene;
using namespace mufflon::scene::geometry;

// Helper macros for error checking and logging
#define FUNCTION_NAME __func__
#define CHECK(x, name, retval)													\
	do {																		\
		if(!x) {																\
			logError("[", FUNCTION_NAME, "] Violated condition (" #name ")");	\
			return retval;														\
		}																		\
	} while(0)
#define CHECK_NULLPTR(x, name, retval)											\
	do {																		\
		if(x == nullptr) {														\
			logError("[", FUNCTION_NAME, "] Invalid " #name " (nullptr)");		\
			return retval;														\
		}																		\
	} while(0)
#define CHECK_GEQ_ZERO(x, name, retval)											\
	do {																		\
		if(x < 0) {																\
			logError("[", FUNCTION_NAME, "] Invalid " #name " (< 0)");			\
			return retval;														\
		}																		\
	} while(0)

// Shortcuts for long handle names
template < class T >
using PolyVAttr = Polygons::VertexAttributeHandle<T>;
template < class T >
using PolyFAttr = Polygons::FaceAttributeHandle<T>;
template < class T >
using SphereAttr = Spheres::AttributeHandle<T>;
using PolyVHdl = Polygons::VertexHandle;
using PolyFHdl = Polygons::FaceHandle;
using SphereVHdl = Spheres::SphereHandle;

// Return values for invalid handles/attributes
namespace {

// static variables for interacting with the renderer
std::unique_ptr<renderer::IRenderer> s_currentRenderer;
std::unique_ptr<renderer::OutputHandler> s_imageOutput;
renderer::OutputValue s_outputTargets;
static void(*s_logCallback)(const char*, int);
// TODO: remove these (leftover from Felix' prototype)
std::string s_lastError;

// Plugin container
std::vector<TextureLoaderPlugin> s_plugins;

constexpr PolygonAttributeHandle INVALID_POLY_VATTR_HANDLE{
	INVALID_INDEX, INVALID_INDEX,
	AttribDesc{
		AttributeType::ATTR_COUNT,
		0u
	},
	false
};
constexpr PolygonAttributeHandle INVALID_POLY_FATTR_HANDLE{
	INVALID_INDEX, INVALID_INDEX,
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
	if(s_logCallback != nullptr)
		s_logCallback(message.c_str(), static_cast<int>(severity));
}

} // namespace

// TODO: remove, Felix prototype
const char* get_error(int& length) {
	length = static_cast<int>(s_lastError.length());
	return s_lastError.data();
}
Boolean iterate() {
	static float t = 0.0f;
	glClearColor(0.0f, t, 0.0f, 1.0f);
	t += 0.01f;
	if(t > 1.0f) t -= 1.0f;
	glClear(GL_COLOR_BUFFER_BIT);

	return true;
}
Boolean display_screenshot() {
	if (!s_imageOutput) {
		logError("[", FUNCTION_NAME, "] No image to print is present");
		return false;
	}

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
		"void main() {"
		"	gl_FragColor.xyz = texture2D(textureSampler, texcoord).rgb;"
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
		if (tex != 0) glDeleteTextures(1u, &tex);
		if (vertShader != 0) glDeleteShader(vertShader);
		if (geomShader != 0) glDeleteShader(geomShader);
		if (fragShader != 0) glDeleteShader(fragShader);
		if (program != 0) glDeleteProgram(program);
		if (vao != 0) glDeleteVertexArrays(1, &vao);
	};

	// Creates an OpenGL texture, copies the screen data into it and
	// renders it to the screen
	glGenTextures(1u, &tex);
	if (tex == 0) {
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
	if (vertShader == 0 || fragShader == 0 || geomShader == 0) {
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
	if (compiled[0] != GL_TRUE || compiled[1] != GL_TRUE || compiled[2] != GL_TRUE) {
		logError("[", FUNCTION_NAME, "] Failed to compile screen shaders");
		cleanup();
		return false;
	}

	program = glCreateProgram();
	if (program == 0u) {
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
	if (linkStatus != GL_TRUE) {
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

	glGenVertexArrays(1, &vao);
	if (vao == 0) {
		logError("[", FUNCTION_NAME, "] Failed to link screen program");
		cleanup();
		return false;
	}
	glBindVertexArray(vao);
	glDrawArrays(GL_POINTS, 0, 1u);

	cleanup();
	return true;
}
Boolean resize(int width, int height, int offsetX, int offsetY) {
	// glViewport should not be called with width or height zero
	if(width == 0 || height == 0) return true;
	glViewport(0, 0, width, height);
	return true;
}
void execute_command(const char* command) {
	// TODO
}



Boolean polygon_resize(ObjectHdl obj, size_t vertices, size_t edges, size_t faces) {
	CHECK_NULLPTR(obj, "object handle", false);
	static_cast<Object*>(obj)->template resize<Polygons>(vertices, edges, faces);
	return true;
}

PolygonAttributeHandle polygon_request_vertex_attribute(ObjectHdl obj, const char* name,
														AttribDesc type) {
	CHECK_NULLPTR(obj, "object handle", INVALID_POLY_VATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_POLY_VATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [name, &type, &object](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		auto attr = object.template request<Polygons, PolyVAttr<Type>>(name);
		return PolygonAttributeHandle{
			attr.omHandle.idx(),
			static_cast<int32_t>(attr.customHandle.index()),
			type, false
		};
	}, [&type, name = FUNCTION_NAME](){
		logError("[", name, "] Unknown/Unsupported attribute type ", get_attr_type_name(type));
		return INVALID_POLY_VATTR_HANDLE;
	});
}

PolygonAttributeHandle polygon_request_face_attribute(ObjectHdl obj,
													  const char* name,
													  AttribDesc type) {
	CHECK_NULLPTR(obj, "object handle", INVALID_POLY_FATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_POLY_FATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [name, type, &object](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		auto attr = object.template request<Polygons, PolyFAttr<Type>>(name);
		return PolygonAttributeHandle{
			attr.omHandle.idx(),
			static_cast<int32_t>(attr.customHandle.index()),
			type, true
		};
	}, [&type, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type", get_attr_type_name(type));
		return INVALID_POLY_FATTR_HANDLE;
	});
}

Boolean polygon_remove_vertex_attribute(ObjectHdl obj, const PolygonAttributeHandle* hdl) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(hdl, "attribute", false);
	CHECK_GEQ_ZERO(hdl->openMeshIndex, "attribute index (OpenMesh)", false);
	CHECK_GEQ_ZERO(hdl->customIndex, "attribute index (custom)", false);
	CHECK(!hdl->face, "face attribute in vertex function", false);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(hdl->type, [hdl, &object](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		PolyVAttr<Type> attr = convert_poly_to_attr<PolyVAttr<Type>>(*hdl);
		object.template remove<Polygons>(attr);
		return true;
	}, [hdl, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(hdl->type));
		return false;
	});
}

Boolean polygon_remove_face_attribute(ObjectHdl obj, const PolygonAttributeHandle* hdl) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(hdl, "attribute", false);
	CHECK_GEQ_ZERO(hdl->openMeshIndex, "attribute index (OpenMesh)", false);
	CHECK_GEQ_ZERO(hdl->customIndex, "attribute index (custom)", false);
	CHECK(hdl->face, "vertex attribute in face function", false);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(hdl->type, [hdl, &object](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		PolyFAttr<Type> attr = convert_poly_to_attr<PolyFAttr<Type>>(*hdl);
		object.template remove<Polygons>(attr);
		return true;
	}, [hdl, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(hdl->type));
		return false;
	});
}

PolygonAttributeHandle polygon_find_vertex_attribute(ObjectHdl obj,
													 const char* name,
													 AttribDesc type) {
	CHECK_NULLPTR(obj, "object handle", INVALID_POLY_VATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_POLY_VATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [&object, type, name](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		std::optional<PolyVAttr<Type>> attr = object.template find<Polygons, PolyVAttr<Type>>(name);
		if(attr.has_value()) {
			return PolygonAttributeHandle{
				attr.value().omHandle.idx(),
				static_cast<int32_t>(attr.value().customHandle.index()),
				type, false
			};
		}
		return INVALID_POLY_VATTR_HANDLE;
	}, [&type, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type", get_attr_type_name(type));
		return INVALID_POLY_VATTR_HANDLE;
	});
}

PolygonAttributeHandle polygon_find_face_attribute(ObjectHdl obj, const char* name,
												   AttribDesc type) {
	CHECK_NULLPTR(obj, "object handle", INVALID_POLY_FATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_POLY_FATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [&object, type, name](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		std::optional<PolyFAttr<Type>> attr = object.template find<Polygons, PolyFAttr<Type>>(name);
		if(attr.has_value()) {
			return PolygonAttributeHandle{
				attr.value().omHandle.idx(),
				static_cast<int32_t>(attr.value().customHandle.index()),
				type, true
			};
		}
		return INVALID_POLY_FATTR_HANDLE;
	}, [&type, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type", get_attr_type_name(type));
		return INVALID_POLY_FATTR_HANDLE;
	});
}

VertexHdl polygon_add_vertex(ObjectHdl obj, Vec3 point, Vec3 normal, Vec2 uv) {
	CHECK_NULLPTR(obj, "object handle", VertexHdl{ INVALID_INDEX });
	PolyVHdl hdl = static_cast<Object*>(obj)->template add<Polygons>(
		util::pun<ei::Vec3>(point), util::pun<ei::Vec3>(normal),
		util::pun<ei::Vec2>(uv));
	if(!hdl.is_valid()) {
		logError("[", FUNCTION_NAME, "] Error adding vertex to polygon");
		return VertexHdl{ INVALID_INDEX };
	}
	return VertexHdl{ static_cast<IndexType>(hdl.idx()) };
}

FaceHdl polygon_add_triangle(ObjectHdl obj, UVec3 vertices) {
	CHECK_NULLPTR(obj, "object handle", FaceHdl{ INVALID_INDEX });

	PolyFHdl hdl = static_cast<Object*>(obj)->template add<Polygons>(
		PolyVHdl{ static_cast<int>(vertices.x) },
		PolyVHdl{ static_cast<int>(vertices.y) },
		PolyVHdl{ static_cast<int>(vertices.z) });
	if(!hdl.is_valid()) {
		logError("[", FUNCTION_NAME, "] Error adding triangle to polygon");
		return FaceHdl{ INVALID_INDEX };
	}
	return FaceHdl{ static_cast<IndexType>(hdl.idx()) };
}

FaceHdl polygon_add_triangle_material(ObjectHdl obj, UVec3 vertices,
										 MatIdx idx) {
	CHECK_NULLPTR(obj, "object handle", FaceHdl{ INVALID_INDEX });
	PolyFHdl hdl = static_cast<Object*>(obj)->template add<Polygons>(
		PolyVHdl{ static_cast<int>(vertices.x) },
		PolyVHdl{ static_cast<int>(vertices.y) },
		PolyVHdl{ static_cast<int>(vertices.z) },
		MaterialIndex{ idx });
	if(!hdl.is_valid()) {
		logError("[", FUNCTION_NAME, "] Error adding triangle to polygon");
		return FaceHdl{ INVALID_INDEX };
	}
	return FaceHdl{ static_cast<IndexType>(hdl.idx()) };
}

FaceHdl polygon_add_quad(ObjectHdl obj, UVec4 vertices) {
	CHECK_NULLPTR(obj, "object handle", FaceHdl{ INVALID_INDEX });
	PolyFHdl hdl = static_cast<Object*>(obj)->template add<Polygons>(
		PolyVHdl{ static_cast<int>(vertices.x) },
		PolyVHdl{ static_cast<int>(vertices.y) },
		PolyVHdl{ static_cast<int>(vertices.z) },
		PolyVHdl{ static_cast<int>(vertices.w) });
	if(!hdl.is_valid()) {
		logError("[", FUNCTION_NAME, "] Error adding triangle to polygon");
		return FaceHdl{ INVALID_INDEX };
	}
	return FaceHdl{ static_cast<IndexType>(hdl.idx()) };
}

FaceHdl polygon_add_quad_material(ObjectHdl obj, UVec4 vertices,
									 MatIdx idx) {
	CHECK_NULLPTR(obj, "object handle", FaceHdl{ INVALID_INDEX });
	PolyFHdl hdl = static_cast<Object*>(obj)->template add<Polygons>(
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
}

VertexHdl polygon_add_vertex_bulk(ObjectHdl obj, size_t count, FILE* points,
									 FILE* normals, FILE* uvs,
									 size_t* pointsRead, size_t* normalsRead,
									 size_t* uvsRead) {
	CHECK_NULLPTR(obj, "object handle", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(points, "points stream descriptor", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(normals, "normals stream descriptor", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(uvs, "UV coordinates stream descriptor", VertexHdl{ INVALID_INDEX });
	Object& object = *static_cast<Object*>(obj);
	mufflon::util::FileReader pointReader{ points };
	mufflon::util::FileReader normalReader{ normals };
	mufflon::util::FileReader uvReader{ uvs };

	Polygons::VertexBulkReturn info = object.template add_bulk<Polygons>(count, pointReader,
																normalReader, uvReader);
	if(pointsRead != nullptr)
		*pointsRead = info.readPoints;
	if(pointsRead != nullptr)
		*normalsRead = info.readNormals;
	if(pointsRead != nullptr)
		*uvsRead = info.readUvs;
	return VertexHdl{ static_cast<IndexType>(info.handle.idx()) };
}

VertexHdl polygon_add_vertex_bulk_no_normals(ObjectHdl obj, size_t count,
											 FILE* points, FILE* uvs,
											 size_t* pointsRead,
											 size_t* uvsRead) {
	CHECK_NULLPTR(obj, "object handle", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(points, "points stream descriptor", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(uvs, "UV coordinates stream descriptor", VertexHdl{ INVALID_INDEX });
	Object& object = *static_cast<Object*>(obj);
	mufflon::util::FileReader pointReader{ points };
	mufflon::util::FileReader uvReader{ uvs };

	Polygons::VertexBulkReturn info = object.template add_bulk<Polygons>(count, pointReader,
																		 uvReader);
	if(pointsRead != nullptr)
		*pointsRead = info.readPoints;
	if(pointsRead != nullptr)
		*uvsRead = info.readUvs;
	return VertexHdl{ static_cast<IndexType>(info.handle.idx()) };
}

VertexHdl polygon_add_vertex_bulk_aabb(ObjectHdl obj, size_t count, FILE* points,
										  FILE* normals, FILE* uvs, Vec3 min,
										  Vec3 max, size_t* pointsRead,
										  size_t* normalsRead, size_t* uvsRead) {
	CHECK_NULLPTR(obj, "object handle", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(points, "points stream descriptor", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(normals, "normals stream descriptor", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(uvs, "UV coordinates stream descriptor", VertexHdl{ INVALID_INDEX });
	Object& object = *static_cast<Object*>(obj);
	mufflon::util::FileReader pointReader{ points };
	mufflon::util::FileReader normalReader{ normals };
	mufflon::util::FileReader uvReader{ uvs };

	ei::Box aabb{ util::pun<ei::Vec3>(min), util::pun<ei::Vec3>(max) };
	Polygons::VertexBulkReturn info = object.template add_bulk<Polygons>(count, pointReader,
																normalReader, uvReader,
																aabb);
	if(pointsRead != nullptr)
		*pointsRead = info.readPoints;
	if(pointsRead != nullptr)
		*normalsRead = info.readNormals;
	if(pointsRead != nullptr)
		*uvsRead = info.readUvs;
	return VertexHdl{ static_cast<IndexType>(info.handle.idx()) };
}

VertexHdl polygon_add_vertex_bulk_aabb_no_normals(ObjectHdl obj, size_t count,
												  FILE* points, FILE* uvs,
												  Vec3 min, Vec3 max,
												  size_t* pointsRead,
												  size_t* uvsRead) {
	CHECK_NULLPTR(obj, "object handle", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(points, "points stream descriptor", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(uvs, "UV coordinates stream descriptor", VertexHdl{ INVALID_INDEX });
	Object& object = *static_cast<Object*>(obj);
	mufflon::util::FileReader pointReader{ points };
	mufflon::util::FileReader uvReader{ uvs };

	ei::Box aabb{ util::pun<ei::Vec3>(min), util::pun<ei::Vec3>(max) };
	Polygons::VertexBulkReturn info = object.template add_bulk<Polygons>(count, pointReader,
																		 uvReader, aabb);
	if(pointsRead != nullptr)
		*pointsRead = info.readPoints;
	if(pointsRead != nullptr)
		*uvsRead = info.readUvs;
	return VertexHdl{ static_cast<IndexType>(info.handle.idx()) };
}

Boolean polygon_set_vertex_attribute(ObjectHdl obj, const PolygonAttributeHandle* attr,
								  VertexHdl vertex, const void* value) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(attr, "attribute handle", false);
	CHECK_NULLPTR(value, "attribute value", false);
	CHECK(!attr->face, "Face attribute in vertex function", false);
	CHECK_GEQ_ZERO(vertex, "vertex index", false);
	CHECK_GEQ_ZERO(attr->openMeshIndex, "attribute index (OpenMesh)", false);
	CHECK_GEQ_ZERO(attr->customIndex, "attribute index (custom)", false);
	Object& object = *static_cast<Object*>(obj);
	if(vertex >= static_cast<int>(object.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 vertex, " >= ", object.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return false;
	}

	return switchAttributeType(attr->type, [&object, attr, vertex, value](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		auto& attribute = object.template aquire<Polygons>(convert_poly_to_attr<PolyVAttr<Type>>(*attr));
		(*attribute.template aquire<Device::CPU>())[vertex] = *static_cast<const Type*>(value);
		return true;
	}, [attr, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(attr->type));
		return false;
	});
}

Boolean polygon_set_vertex_normal(ObjectHdl obj, VertexHdl vertex, Vec3 normal) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_GEQ_ZERO(vertex, "vertex index", false);
	Object& object = *static_cast<Object*>(obj);
	if(vertex >= static_cast<int>(object.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 vertex, " >= ", object.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return false;
	}

	(*object.get_geometry<geometry::Polygons>().get_normals().aquire<>())[vertex] = util::pun<OpenMesh::Vec3f>(normal);
	return true;
}

Boolean polygon_set_vertex_uv(ObjectHdl obj, VertexHdl vertex, Vec2 uv) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_GEQ_ZERO(vertex, "vertex index", false);
	Object& object = *static_cast<Object*>(obj);
	if(vertex >= static_cast<int>(object.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 vertex, " >= ", object.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return false;
	}

	(*object.get_geometry<geometry::Polygons>().get_uvs().aquire<>())[vertex] = util::pun<OpenMesh::Vec2f>(uv);
	return true;
}

Boolean polygon_set_face_attribute(ObjectHdl obj, const PolygonAttributeHandle* attr,
								FaceHdl face, const void* value) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(attr, "attribute handle", false);
	CHECK_NULLPTR(value, "attribute value", false);
	CHECK(attr->face, "Vertex attribute in face function", false);
	CHECK_GEQ_ZERO(face, "face index", false);
	CHECK_GEQ_ZERO(attr->openMeshIndex, "attribute index (OpenMesh)", false);
	CHECK_GEQ_ZERO(attr->customIndex, "attribute index (custom)", false);
	Object& object = *static_cast<Object*>(obj);
	if(face >= static_cast<int>(object.template get_geometry<Polygons>().get_face_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 face, " >= ", object.template get_geometry<Polygons>().get_face_count(),
				 ")");
		return false;
	}

	return switchAttributeType(attr->type, [&object, attr, face, value](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		auto& attribute = object.template aquire<Polygons>(convert_poly_to_attr<PolyFAttr<Type>>(*attr));
		(*attribute.template aquire<Device::CPU>())[face] = *static_cast<const Type*>(value);
		return true;
	}, [attr, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(attr->type));
		return false;
	});
}

Boolean polygon_set_material_idx(ObjectHdl obj, FaceHdl face, MatIdx idx) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_GEQ_ZERO(face, "face index", false);
	Object& object = *static_cast<Object*>(obj);
	if(face >= static_cast<int>(object.template get_geometry<Polygons>().get_face_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 face, " >= ", object.template get_geometry<Polygons>().get_face_count(),
				 ")");
		return false;
	}

	(*object.template get_mat_indices<Polygons>().aquire())[face] = idx;
	return true;
}

size_t polygon_set_vertex_attribute_bulk(ObjectHdl obj, const PolygonAttributeHandle* attr,
										 VertexHdl startVertex, size_t count,
										 FILE* stream) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	CHECK_NULLPTR(attr, "attribute handle", INVALID_SIZE);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK(!attr->face, "Face attribute in vertex function", false);
	CHECK_GEQ_ZERO(startVertex, "start vertex index", INVALID_SIZE);
	CHECK_GEQ_ZERO(attr->openMeshIndex, "attribute index (OpenMesh)", INVALID_SIZE);
	CHECK_GEQ_ZERO(attr->customIndex, "attribute index (custom)", INVALID_SIZE);
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
		return object.template add_bulk<Polygons>(convert_poly_to_attr<PolyVAttr<Type>>(*attr),
										 PolyVHdl{ static_cast<int>(startVertex) },
										 count, attrStream);
	}, [attr, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(attr->type));
		return INVALID_SIZE;
	});
}

size_t polygon_set_face_attribute_bulk(ObjectHdl obj, const PolygonAttributeHandle* attr,
									   FaceHdl startFace, size_t count,
									   FILE* stream) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	CHECK_NULLPTR(attr, "attribute handle", INVALID_SIZE);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK(attr->face, "Vertex attribute in face function", false);
	CHECK_GEQ_ZERO(startFace, "start face index", INVALID_SIZE);
	CHECK_GEQ_ZERO(attr->openMeshIndex, "attribute index (OpenMesh)", INVALID_SIZE);
	CHECK_GEQ_ZERO(attr->customIndex, "attribute index (custom)", INVALID_SIZE);
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
		return object.template add_bulk<Polygons>(convert_poly_to_attr<PolyFAttr<Type>>(*attr),
										 PolyFHdl{ static_cast<int>(startFace) },
										 count, attrStream);
	}, [attr, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(attr->type));
		return INVALID_SIZE;
	});
}

size_t polygon_set_material_idx_bulk(ObjectHdl obj, FaceHdl startFace, size_t count,
									 FILE* stream) {
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

	return object.template add_bulk<Polygons>(object.template get_mat_indices<Polygons>(),
									 PolyFHdl{ static_cast<int>(startFace) },
									 count, matStream);
}

size_t polygon_get_vertex_count(ObjectHdl obj) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.template get_geometry<Polygons>().get_vertex_count();
}

size_t polygon_get_edge_count(ObjectHdl obj) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.template get_geometry<Polygons>().get_edge_count();
}

size_t polygon_get_face_count(ObjectHdl obj) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.template get_geometry<Polygons>().get_face_count();
}

size_t polygon_get_triangle_count(ObjectHdl obj) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.template get_geometry<Polygons>().get_triangle_count();
}

size_t polygon_get_quad_count(ObjectHdl obj) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.template get_geometry<Polygons>().get_quad_count();
}

Boolean polygon_get_bounding_box(ObjectHdl obj, Vec3* min, Vec3* max) {
	CHECK_NULLPTR(obj, "object handle", false);
	const Object& object = *static_cast<const Object*>(obj);
	const ei::Box& aabb = object.template get_geometry<Polygons>().get_bounding_box();
	if(min != nullptr)
		*min = util::pun<Vec3>(aabb.min);
	if(max != nullptr)
		*max = util::pun<Vec3>(aabb.max);
	return true;
}

Boolean spheres_resize(ObjectHdl obj, size_t count) {
	CHECK_NULLPTR(obj, "object handle", false);
	static_cast<Object*>(obj)->template resize<Spheres>(count);
	return true;
}

SphereAttributeHandle spheres_request_attribute(ObjectHdl obj, const char* name,
												AttribDesc type) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SPHERE_ATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_SPHERE_ATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [name, type, &object](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		return SphereAttributeHandle{
			static_cast<int>(object.template request<Spheres, Type>(name).index()),
			type
		};
	}, [&type, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type", get_attr_type_name(type));
		return INVALID_SPHERE_ATTR_HANDLE;
	});
}

Boolean spheres_remove_attribute(ObjectHdl obj, const SphereAttributeHandle* hdl) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(hdl, "attribute", false);
	CHECK_GEQ_ZERO(hdl->index, "attribute index", false);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(hdl->type, [hdl, &object](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		SphereAttr<Type> attr{ static_cast<size_t>(hdl->index) };
		object.template remove<Spheres>(attr);
		return true;
	}, [hdl, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(hdl->type));
		return false;
	});
}

SphereAttributeHandle spheres_find_attribute(ObjectHdl obj, const char* name,
											 AttribDesc type) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SPHERE_ATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_SPHERE_ATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [&object, type, name](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		std::optional<SphereAttr<Type>> attr = object.template find<Spheres, Type>(name);
		if(attr.has_value()) {
			return SphereAttributeHandle{
				static_cast<IndexType>(attr.value().index()),
				type
			};
		}
		return INVALID_SPHERE_ATTR_HANDLE;
	}, [&type, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type", get_attr_type_name(type));
		return INVALID_SPHERE_ATTR_HANDLE;
	});
}

SphereHdl spheres_add_sphere(ObjectHdl obj, Vec3 point, float radius) {
	CHECK_NULLPTR(obj, "object handle", SphereHdl{ INVALID_INDEX });
	SphereVHdl hdl = static_cast<Object*>(obj)->template add<Spheres>(
		util::pun<ei::Vec3>(point), radius);
	return SphereHdl{ static_cast<IndexType>(hdl) };
}

SphereHdl spheres_add_sphere_material(ObjectHdl obj, Vec3 point, float radius,
								MatIdx idx) {
	CHECK_NULLPTR(obj, "object handle", SphereHdl{ INVALID_INDEX });
	SphereVHdl hdl = static_cast<Object*>(obj)->template add<Spheres>(
		util::pun<ei::Vec3>(point), radius, idx);
	return SphereHdl{ static_cast<IndexType>(hdl) };
}

SphereHdl spheres_add_sphere_bulk(ObjectHdl obj, size_t count,
									 FILE* stream, size_t* readSpheres) {
	CHECK_NULLPTR(obj, "object handle", SphereHdl{ INVALID_INDEX } );
	CHECK_NULLPTR(stream, "sphere stream descriptor", SphereHdl{ INVALID_INDEX });
	Object& object = *static_cast<Object*>(obj);
	mufflon::util::FileReader sphereReader{ stream };

	Spheres::BulkReturn info = object.template add_bulk<Spheres>(count, sphereReader);
	if(readSpheres != nullptr)
		*readSpheres = info.readSpheres;
	return SphereHdl{ static_cast<IndexType>(info.handle) };
}

SphereHdl spheres_add_sphere_bulk_aabb(ObjectHdl obj, size_t count,
										  FILE* stream, Vec3 min, Vec3 max,
										  size_t* readSpheres) {

	CHECK_NULLPTR(obj, "object handle", SphereHdl{ INVALID_INDEX });
	CHECK_NULLPTR(stream, "sphere stream descriptor", SphereHdl{ INVALID_INDEX });
	Object& object = *static_cast<Object*>(obj);
	mufflon::util::FileReader sphereReader{ stream };

	ei::Box aabb{ util::pun<ei::Vec3>(min), util::pun<ei::Vec3>(max) };
	Spheres::BulkReturn info = object.template add_bulk<Spheres>(count, sphereReader,
														aabb);
	if(readSpheres != nullptr)
		*readSpheres = info.readSpheres;
	return SphereHdl{ static_cast<IndexType>(info.handle) };
}

Boolean spheres_set_attribute(ObjectHdl obj, const SphereAttributeHandle* attr,
						   SphereHdl sphere, const void* value) {
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
		SphereAttr<Type> sphereAttr{ static_cast<size_t>(attr->index) };
		auto& attribute = object.template aquire<Spheres>(sphereAttr);
		(*attribute.template aquire<Device::CPU>())[sphere] = *static_cast<const Type*>(value);
		return true;
	}, [attr, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(attr->type));
		return false;
	});
}

Boolean spheres_set_material_idx(ObjectHdl obj, SphereHdl sphere, MatIdx idx) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_GEQ_ZERO(sphere, "sphere index", false);
	Object& object = *static_cast<Object*>(obj);
	if(sphere >= static_cast<int>(object.template get_geometry<Spheres>().get_sphere_count())) {
		logError("[", FUNCTION_NAME, "] Sphere index out of bounds (",
				 sphere, " >= ", object.template get_geometry<Spheres>().get_sphere_count(),
				 ")");
		return false;
	}

	(*object.template get_mat_indices<Spheres>().aquire())[sphere] = idx;
	return true;
}

size_t spheres_set_attribute_bulk(ObjectHdl obj, const SphereAttributeHandle* attr,
								  SphereHdl startSphere, size_t count,
								  FILE* stream) {
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
		SphereAttr<Type> sphereAttr{ static_cast<size_t>(attr->index) };
		return object.template add_bulk<Spheres>(sphereAttr,
										SphereVHdl{ static_cast<size_t>(startSphere) },
										count, attrStream);
	}, [attr, name = FUNCTION_NAME]() {
		logError("[", name, "] Unknown/Unsupported attribute type",
				 get_attr_type_name(attr->type));
		return INVALID_SIZE;
	});
}

size_t spheres_set_material_idx_bulk(ObjectHdl obj, SphereHdl startSphere, size_t count,
									 FILE* stream) {
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

	return object.template add_bulk<Spheres>(object.template get_mat_indices<Spheres>(),
									SphereVHdl{ static_cast<size_t>(startSphere) },
									count, matStream);
	return INVALID_SIZE;
}

size_t spheres_get_sphere_count(ObjectHdl obj) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.template get_geometry<Spheres>().get_sphere_count();
}

Boolean spheres_get_bounding_box(ObjectHdl obj, Vec3* min, Vec3* max) {
	CHECK_NULLPTR(obj, "object handle", false);
	const Object& object = *static_cast<const Object*>(obj);
	const ei::Box& aabb = object.get_bounding_box<Spheres>();
	if(min != nullptr)
		*min = util::pun<Vec3>(aabb.min);
	if(max != nullptr)
		*max = util::pun<Vec3>(aabb.max);
	return true;
}

Boolean instance_set_transformation_matrix(InstanceHdl inst, const Mat4x3* mat) {
	CHECK_NULLPTR(inst, "instance handle", false);
	CHECK_NULLPTR(mat, "transformation matrix", false);
	Instance& instance = *static_cast<InstanceHandle>(inst);
	instance.set_transformation_matrix(util::pun<ei::Matrix<float, 4u, 3u>>(*mat));
	return true;
}

Boolean instance_get_transformation_matrix(InstanceHdl inst, Mat4x3* mat) {
	CHECK_NULLPTR(inst, "instance handle", false);
	const Instance& instance = *static_cast<ConstInstanceHandle>(inst);
	if(mat != nullptr)
		*mat = util::pun<Mat4x3>(instance.get_transformation_matrix());
	return true;
}

Boolean instance_get_bounding_box(InstanceHdl inst, Vec3* min, Vec3* max) {
	CHECK_NULLPTR(inst, "instance handle", false);
	const Instance& instance = *static_cast<ConstInstanceHandle>(inst);
	const ei::Box& aabb = instance.get_bounding_box();
	if(min != nullptr)
		*min = util::pun<Vec3>(aabb.min);
	if(max != nullptr)
		*max = util::pun<Vec3>(aabb.max);
	return true;
}

ObjectHdl world_create_object(const char* name) {
	CHECK_NULLPTR(name, "object name", nullptr);
	return static_cast<ObjectHdl>(WorldContainer::instance().create_object(name));
}

ObjectHdl world_get_object(const char* name) {
	CHECK_NULLPTR(name, "object name", nullptr);
	return static_cast<ObjectHdl>(WorldContainer::instance().get_object(name));
}

InstanceHdl world_create_instance(ObjectHdl obj) {
	CHECK_NULLPTR(obj, "object handle", nullptr);
	ObjectHandle hdl = static_cast<Object*>(obj);
	return static_cast<InstanceHdl>(WorldContainer::instance().create_instance(hdl));
}

ScenarioHdl world_create_scenario(const char* name) {
	CHECK_NULLPTR(name, "scenario name", nullptr);
	ScenarioHandle hdl = WorldContainer::instance().create_scenario(name);
	return static_cast<ScenarioHdl>(hdl);
}

ScenarioHdl world_find_scenario(const char* name) {
	CHECK_NULLPTR(name, "scenario name", nullptr);
	std::string_view nameView{ name };
	ScenarioHandle hdl = WorldContainer::instance().get_scenario(nameView);
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Could not find scenario '",
				 nameView, "'");
		return nullptr;
	}
	return static_cast<ScenarioHdl>(hdl);
}

MaterialHdl world_add_material(const char* name, const MaterialParams* mat) {
	CHECK_NULLPTR(name, "material name", nullptr);
	CHECK_NULLPTR(mat, "material parameters", nullptr);

	MaterialHandle hdl = nullptr;
	switch(mat->innerType) {
		case MATERIAL_LAMBERT: {
			auto tex = WorldContainer::instance().add_texture(textures::Format::RGB32F, mat->inner.lambert.albedo.rgb);
			hdl = WorldContainer::instance().add_material(std::make_unique<materials::Lambert>(&tex->second));
		}	break;
		case MATERIAL_LAMBERT_TEXTURED: {
			auto tex = mat->inner.lambert.albedo.tex;
			hdl = WorldContainer::instance().add_material(std::make_unique<materials::Lambert>(static_cast<TextureHandle>(tex)));
		}	break;
		case MATERIAL_TORRANCE:
		case MATERIAL_TORRANCE_TEXALBEDO:
		case MATERIAL_TORRANCE_ANISOTROPIC:
		case MATERIAL_TORRANCE_ANISOTROPIC_TEXALBEDO:
		case MATERIAL_TORRANCE_TEXTURED:
		case MATERIAL_TORRANCE_TEXTURED_TEXALBEDO:
			// TODO
			logWarning("[", FUNCTION_NAME, "] Material type 'torrance' not supported yet");
			return nullptr;
		case MATERIAL_WALTER:
		case MATERIAL_WALTER_ANISOTROPIC:
		case MATERIAL_WALTER_TEXTURED:
			logWarning("[", FUNCTION_NAME, "] Material type 'walter' not supported yet");
			return nullptr;
		case MATERIAL_EMISSIVE:
		case MATERIAL_EMISSIVE_TEXTURED:
			logWarning("[", FUNCTION_NAME, "] Material type 'emissive' not supported yet");
			return nullptr;
		case MATERIAL_ORENNAYAR:
		case MATERIAL_ORENNAYAR_TEXTURED:
			logWarning("[", FUNCTION_NAME, "] Material type 'orennayar' not supported yet");
			return nullptr;
		case MATERIAL_BLEND:
			logWarning("[", FUNCTION_NAME, "] Material type 'blend' not supported yet");
			return nullptr;
		case MATERIAL_FRESNEL:
		case MATERIAL_FRESNEL_COMPLEX:
			logWarning("[", FUNCTION_NAME, "] Material type 'fresnel' not supported yet");
			return nullptr;
		default:
			logWarning("[", FUNCTION_NAME, "] Unknown material type");
	}

	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Error creating material '",
				 name, "'");
		return nullptr;
	}

	return static_cast<MaterialHdl>(hdl);
}

CameraHdl world_add_pinhole_camera(const char* name, Vec3 position, Vec3 dir,
								   Vec3 up, float near, float far, float vFov) {
	CHECK_NULLPTR(name, "camera name", nullptr);
	CameraHandle hdl = WorldContainer::instance().add_camera(name,
		std::make_unique<cameras::Pinhole>(
			util::pun<ei::Vec3>(position), util::pun<ei::Vec3>(dir),
			util::pun<ei::Vec3>(up), vFov, near, far
	));
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Error creating pinhole camera");
		return nullptr;
	}
	return static_cast<CameraHdl>(hdl);
}

LightHdl world_add_point_light(const char* name, Vec3 position, Vec3 intensity) {
	CHECK_NULLPTR(name, "pointlight name", nullptr);
	auto hdl = WorldContainer::instance().add_light(name, lights::PointLight{
													util::pun<ei::Vec3>(position),
													util::pun<ei::Vec3>(intensity) });
	if(!hdl.has_value()) {
		logError("[", FUNCTION_NAME, "] Error adding point light");
		return nullptr;
	}
	return static_cast<LightHdl>(&hdl.value()->second);
}

LightHdl world_add_spot_light(const char* name, Vec3 position, Vec3 direction,
							  Vec3 intensity, float openingAngle,
							  float falloffStart) {
	CHECK_NULLPTR(name, "spotlight name", nullptr);
	ei::Vec3 dir = ei::normalize(util::pun<ei::Vec3>(direction));
	if(!ei::approx(ei::len(dir), 1.0f)) {
		logError("[", FUNCTION_NAME, "] Spotlight direction cannot be a null vector");
		return nullptr;
	}
	float actualAngle = std::fmod(openingAngle, 2.f * ei::PI);
	float actualFalloff = std::fmod(falloffStart, 2.f * ei::PI);
	if(actualAngle < 0.f)
		actualAngle += 2.f*ei::PI;
	if(actualFalloff < 0.f)
		actualFalloff += 2.f*ei::PI;
	if(actualAngle > ei::PI / 2.f) {
		logWarning("[", FUNCTION_NAME, "] Spotlight angle will be clamped between 0-180 degrees");
		actualAngle = ei::PI / 2.f;
	}
	float cosAngle = std::cos(actualAngle);
	float cosFalloff = std::cos(actualFalloff);
	if(cosAngle > cosFalloff) {
		logWarning("[", FUNCTION_NAME, "] Spotlight falloff angle cannot be larger than"
				   " its opening angle");
		cosFalloff = cosAngle;
	}

	auto hdl = WorldContainer::instance().add_light(name, lights::SpotLight{
													util::pun<ei::Vec3>(position),
													ei::packOctahedral32(dir),
													util::pun<ei::Vec3>(intensity),
													__float2half(cosAngle),
													__float2half(cosFalloff) });
	if(!hdl.has_value()) {
		logError("[", FUNCTION_NAME, "] Error adding point light");
		return nullptr;
	}
	return static_cast<LightHdl>(&hdl.value()->second);
}

LightHdl world_add_directional_light(const char* name, Vec3 direction,
									 Vec3 radiance) {
	CHECK_NULLPTR(name, "directional light name", nullptr);
	auto hdl = WorldContainer::instance().add_light(name, lights::DirectionalLight{
													util::pun<ei::Vec3>(direction),
													util::pun<ei::Vec3>(radiance) });
	if(!hdl.has_value()) {
		logError("[", FUNCTION_NAME, "] Error adding directional light");
		return nullptr;
	}
	return static_cast<LightHdl>(&hdl.value()->second);
}

LightHdl world_add_envmap_light(const char* name, TextureHdl envmap) {
	CHECK_NULLPTR(name, "directional light name", nullptr);
	CHECK_NULLPTR(envmap, "environment map", nullptr);
	TextureHandle hdl = static_cast<TextureHandle>(envmap);
	auto envLight = WorldContainer::instance().add_light(name, hdl);
	if(!envLight.has_value()) {
		logError("[", FUNCTION_NAME, "] Error adding environment-map light");
		return nullptr;
	}
	return envLight.value()->second;
}

CameraHdl world_get_camera(const char* name) {
	CHECK_NULLPTR(name, "camera name", nullptr);
	return static_cast<CameraHdl>(WorldContainer::instance().get_camera(name));
}
LightHdl world_get_light(const char* name, LightType type) {
	CHECK_NULLPTR(name, "light name", nullptr);
	switch(type) {
		case LightType::LIGHT_POINT: {
			auto hdl = WorldContainer::instance().get_point_light(name);
			if(!hdl.has_value()) {
				logError("[", FUNCTION_NAME, "] Error getting point light");
				return nullptr;
			}
			return static_cast<LightHdl>(&hdl.value()->second);
		}
		case LightType::LIGHT_SPOT: {
			auto hdl = WorldContainer::instance().get_spot_light(name);
			if(!hdl.has_value()) {
				logError("[", FUNCTION_NAME, "] Error getting spot light");
				return nullptr;
			}
			return static_cast<LightHdl>(&hdl.value()->second);
		}
		case LightType::LIGHT_DIRECTIONAL: {
			auto hdl = WorldContainer::instance().get_dir_light(name);
			if(!hdl.has_value()) {
				logError("[", FUNCTION_NAME, "] Error getting directional light");
				return nullptr;
			}
			return static_cast<LightHdl>(&hdl.value()->second);
		}
		case LightType::LIGHT_ENVMAP: {
			auto hdl = WorldContainer::instance().get_env_light(name);
			if(!hdl.has_value()) {
				logError("[", FUNCTION_NAME, "] Error geting envmap light");
				return nullptr;
			}
			return static_cast<LightHdl>(&hdl.value()->second);
		}
		default: {
			logError("[", FUNCTION_NAME, "] Unknown light type");
			return nullptr;
		}
	}
}

SceneHdl world_load_scenario(ScenarioHdl scenario) {
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	SceneHandle hdl = WorldContainer::instance().load_scene(static_cast<ConstScenarioHandle>(scenario));
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Failed to load scenario");
		return nullptr;
	}
	return static_cast<SceneHdl>(hdl);
}

SceneHdl world_get_current_scene() {
	return static_cast<SceneHdl>(WorldContainer::instance().get_current_scene());
}

Boolean world_exists_texture(const char* path) {
	CHECK_NULLPTR(path, "texture path", false);
	return WorldContainer::instance().has_texture(path);
}

TextureHdl world_get_texture(const char* path) {
	CHECK_NULLPTR(path, "texture path", false);
	auto hdl = WorldContainer::instance().find_texture(path);
	if(!hdl.has_value()) {
		logError("[", FUNCTION_NAME, "] Could not find texture ",
				 path);
		return nullptr;
	}
	return static_cast<TextureHdl>(&hdl.value()->second);
}

TextureHdl world_add_texture(const char* path, TextureSampling sampling,
							 Boolean sRgb) {
	CHECK_NULLPTR(path, "texture path", nullptr);
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
	auto hdl = WorldContainer::instance().add_texture(path, texData.width, texData.height,
													  texData.layers, static_cast<textures::Format>(texData.format),
													  static_cast<textures::SamplingMode>(sampling),
													  sRgb, texData.data);
	return static_cast<TextureHdl>(&hdl->second);
}

const char* scenario_get_name(ScenarioHdl scenario) {
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	// This relies on the fact that the string_view in scenario points to
	// an std::string object, which is null terminated
	return &static_cast<const Scenario*>(scenario)->get_name()[0u];
}

size_t scenario_get_global_lod_level(ScenarioHdl scenario) {
	CHECK_NULLPTR(scenario, "scenario handle", Scenario::NO_CUSTOM_LOD);
	return static_cast<const Scenario*>(scenario)->get_global_lod_level();
}

Boolean scenario_set_global_lod_level(ScenarioHdl scenario, LodLevel level) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	static_cast<Scenario*>(scenario)->set_global_lod_level(level);
	return true;
}

Boolean scenario_get_resolution(ScenarioHdl scenario, uint32_t* width, uint32_t* height) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	ei::IVec2 res = static_cast<const Scenario*>(scenario)->get_resolution();
	if(width != nullptr)
		*width = res.x;
	if(height != nullptr)
		*height = res.y;
	return true;
}

Boolean scenario_set_resolution(ScenarioHdl scenario, uint32_t width, uint32_t height) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	static_cast<Scenario*>(scenario)->set_resolution(ei::IVec2{ width, height });
	return true;
}

CameraHdl scenario_get_camera(ScenarioHdl scenario) {
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	return static_cast<CameraHdl>(static_cast<const Scenario*>(scenario)->get_camera());
}

Boolean scenario_set_camera(ScenarioHdl scenario, CameraHdl cam) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	static_cast<Scenario*>(scenario)->set_camera(static_cast<CameraHandle>(cam));
	return true;
}

Boolean scenario_is_object_masked(ScenarioHdl scenario, ObjectHdl obj) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(obj, "object handle", false);
	return static_cast<const Scenario*>(scenario)->is_masked(static_cast<const Object*>(obj));
}

Boolean scenario_mask_object(ScenarioHdl scenario, ObjectHdl obj) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(obj, "object handle", false);
	static_cast<Scenario*>(scenario)->mask_object(static_cast<const Object*>(obj));
	return true;
}

LodLevel scenario_get_object_lod(ScenarioHdl scenario, ObjectHdl obj) {
	CHECK_NULLPTR(scenario, "scenario handle", Scenario::NO_CUSTOM_LOD);
	CHECK_NULLPTR(obj, "object handle", Scenario::NO_CUSTOM_LOD);
	return static_cast<const Scenario*>(scenario)->get_custom_lod(static_cast<const Object*>(obj));
}

Boolean scenario_set_object_lod(ScenarioHdl scenario, ObjectHdl obj,
							 LodLevel level) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(obj, "object handle", false);
	static_cast<Scenario*>(scenario)->set_custom_lod(static_cast<const Object*>(obj),
													 level);
	return true;
}

IndexType scenario_get_light_count(ScenarioHdl scenario) {
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_INDEX);
	return static_cast<IndexType>(static_cast<const Scenario*>(scenario)->get_light_names().size());
}

const char* scenario_get_light_name(ScenarioHdl scenario, size_t index) {
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	const Scenario& scen = *static_cast<const Scenario*>(scenario);
	if(index >= scen.get_light_names().size()) {
		logError("[", FUNCTION_NAME, "] Light index out of bounds (",
				 index, " >= ", scen.get_light_names().size(), ")");
		return nullptr;
	}
	// The underlying string_view is guaranteed to be null-terminated since it
	// references a std::string object in WorldContainer. Should that ever
	// change we'll need to perform a copy here instead
	return &scen.get_light_names()[index][0u];
}

Boolean scenario_add_light(ScenarioHdl scenario, const char* name) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(name, "light name", false);
	// Indirection via world container because we store string_views
	std::optional<std::string_view> resolvedName = WorldContainer::instance().get_light_name_ref(name);
	if(!resolvedName.has_value()) {
		logError("[", FUNCTION_NAME, "] Light source '", name, "' does not exist");
		return false;
	}
	static_cast<Scenario*>(scenario)->add_light(resolvedName.value());
	return true;
}

Boolean scenario_remove_light_by_index(ScenarioHdl scenario, size_t index) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	Scenario& scen = *static_cast<Scenario*>(scenario);
	if(index >= scen.get_light_names().size()) {
		logError("[", FUNCTION_NAME, "] Light index out of bounds (",
				 index, " >= ", scen.get_light_names().size(), ")");
		return false;
	}
	scen.remove_light(index);
	return true;
}

Boolean scenario_remove_light_by_named(ScenarioHdl scenario, const char* name) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(name, "light name", false);
	Scenario& scen = *static_cast<Scenario*>(scenario);
	scen.remove_light(name);
	return true;
}

MatIdx scenario_declare_material_slot(ScenarioHdl scenario,
									  const char* name, std::size_t nameLength) {
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_MATERIAL);
	CHECK_NULLPTR(name, "material name", INVALID_MATERIAL);
	std::string_view nameView(name, std::min<std::size_t>(nameLength, std::strlen(name)));
	return static_cast<Scenario*>(scenario)->declare_material_slot(nameView);
}

MatIdx scenario_get_material_slot(ScenarioHdl scenario,
								  const char* name, std::size_t nameLength) {
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_MATERIAL);
	CHECK_NULLPTR(name, "material name", INVALID_MATERIAL);
	std::string_view nameView(name, std::min<std::size_t>(nameLength, std::strlen(name)));
	return static_cast<const Scenario*>(scenario)->get_material_slot_index(nameView);
}

MaterialHdl scenario_get_assigned_material(ScenarioHdl scenario,
										   MatIdx index) {
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	CHECK((index < INVALID_MATERIAL), "Invalid material index", nullptr);
	return static_cast<MaterialHdl>(static_cast<const Scenario*>(scenario)->get_assigned_material(index));
}

Boolean scenario_assign_material(ScenarioHdl scenario, MatIdx index,
							  MaterialHdl handle) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(handle, "material handle", false);
	CHECK((index < INVALID_MATERIAL), "Invalid material index", false);
	static_cast<Scenario*>(scenario)->assign_material(index, static_cast<MaterialHandle>(handle));
	return true;
}

Boolean scene_get_bounding_box(SceneHdl scene, Vec3* min, Vec3* max) {
	CHECK_NULLPTR(scene, "scene handle", false);
	const Scene& scen = *static_cast<const Scene*>(scene);
	if(min != nullptr)
		*min = util::pun<Vec3>(scen.get_bounding_box().min);
	if(max != nullptr)
		*max = util::pun<Vec3>(scen.get_bounding_box().max);
	return true;
}

ConstCameraHdl scene_get_camera(SceneHdl scene) {
	CHECK_NULLPTR(scene, "scene handle", nullptr);
	return static_cast<ConstCameraHdl>(static_cast<const Scene*>(scene)->get_camera());
}

Boolean world_get_point_light_position(LightHdl hdl, Vec3* pos) {
	CHECK_NULLPTR(hdl, "pointlight handle", false);
	const lights::PointLight& light = *static_cast<const lights::PointLight*>(hdl);
	if(pos != nullptr)
		*pos = util::pun<Vec3>(light.position);
	return true;
}

Boolean world_get_point_light_intensity(LightHdl hdl, Vec3* intensity) {
	CHECK_NULLPTR(hdl, "pointlight handle", false);
	const lights::PointLight& light = *static_cast<const lights::PointLight*>(hdl);
	if(intensity != nullptr)
		*intensity = util::pun<Vec3>(light.intensity);
	return true;
}

Boolean world_set_point_light_position(LightHdl hdl, Vec3 pos) {
	CHECK_NULLPTR(hdl, "pointlight handle", false);
	lights::PointLight& light = *static_cast<lights::PointLight*>(hdl);
	light.position = util::pun<ei::Vec3>(pos);
	return true;
}

Boolean world_set_point_light_intensity(LightHdl hdl, Vec3 intensity) {
	CHECK_NULLPTR(hdl, "pointlight handle", false);
	lights::PointLight& light = *static_cast<lights::PointLight*>(hdl);
	light.intensity = util::pun<ei::Vec3>(intensity);
	return true;
}

Boolean world_get_spot_light_position(LightHdl hdl, Vec3* pos) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	const lights::SpotLight& light = *static_cast<const lights::SpotLight*>(hdl);
	if(pos != nullptr)
		*pos = util::pun<Vec3>(light.position);
	return true;
}

Boolean world_get_spot_light_intensity(LightHdl hdl, Vec3* intensity) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	const lights::SpotLight& light = *static_cast<const lights::SpotLight*>(hdl);
	if(intensity != nullptr)
		*intensity = util::pun<Vec3>(light.intensity);
	return true;
}

Boolean world_get_spot_light_direction(LightHdl hdl, Vec3* direction) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	const lights::SpotLight& light = *static_cast<const lights::SpotLight*>(hdl);
	if(direction != nullptr)
		*direction = util::pun<Vec3>(ei::unpackOctahedral32(light.direction));
	return true;
}

Boolean world_get_spot_light_angle(LightHdl hdl, float* angle) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	const lights::SpotLight& light = *static_cast<const lights::SpotLight*>(hdl);
	if(angle != nullptr)
		*angle = std::acos(__half2float(light.cosThetaMax));
	return true;
}

Boolean world_get_spot_light_falloff(LightHdl hdl, float* falloff) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	const lights::SpotLight& light = *static_cast<const lights::SpotLight*>(hdl);
	if(falloff != nullptr)
		*falloff = std::acos(__half2float(light.cosFalloffStart));
	return true;
}

Boolean world_set_spot_light_position(LightHdl hdl, Vec3 pos) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	lights::SpotLight& light = *static_cast<lights::SpotLight*>(hdl);
	light.position = util::pun<ei::Vec3>(pos);
	return true;
}

Boolean world_set_spot_light_intensity(LightHdl hdl, Vec3 intensity) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	lights::SpotLight& light = *static_cast<lights::SpotLight*>(hdl);
	light.intensity = util::pun<ei::Vec3>(intensity);
	return true;
}

Boolean world_set_spot_light_direction(LightHdl hdl, Vec3 direction) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	lights::SpotLight& light = *static_cast<lights::SpotLight*>(hdl);
	ei::Vec3 actualDirection = ei::normalize(util::pun<ei::Vec3>(direction));
	if(!ei::approx(ei::len(actualDirection), 1.0f)) {
		logError("[", FUNCTION_NAME, "] Spotlight direction cannot be a null vector");
		return false;
	}
	// TODO: check direction
	light.direction = ei::packOctahedral32(util::pun<ei::Vec3>(actualDirection));
	return true;
}

Boolean world_set_spot_light_angle(LightHdl hdl, float angle) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	lights::SpotLight& light = *static_cast<lights::SpotLight*>(hdl);

	float actualAngle = std::fmod(angle, 2.f * ei::PI);
	if(actualAngle < 0.f)
		actualAngle += 2.f*ei::PI;
	if(actualAngle > ei::PI / 2.f) {
		logWarning("[", FUNCTION_NAME, "] Spotlight angle will be clamped between 0-180 degrees");
		actualAngle = ei::PI / 2.f;
	}
	light.cosThetaMax = __float2half(std::cos(actualAngle));
	return true;
}

Boolean world_set_spot_light_falloff(LightHdl hdl, float falloff) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	lights::SpotLight& light = *static_cast<lights::SpotLight*>(hdl);
	// Clamp it to the opening angle!
	float actualFalloff = std::fmod(falloff, 2.f * ei::PI);
	if(actualFalloff < 0.f)
		actualFalloff += 2.f*ei::PI;
	const float cosFalloff = std::cos(actualFalloff);
	const __half compressedCosFalloff = __float2half(cosFalloff);
	if(__half2float(light.cosThetaMax) > __half2float(compressedCosFalloff)) {
		logWarning("[", FUNCTION_NAME, "] Spotlight falloff angle cannot be larger than"
				   " its opening angle");
		light.cosFalloffStart = light.cosThetaMax;
	} else {
		light.cosFalloffStart = compressedCosFalloff;
	}
	return true;
}

Boolean world_get_dir_light_direction(LightHdl hdl, Vec3* direction) {
	CHECK_NULLPTR(hdl, "directional light handle", false);
	const lights::DirectionalLight& light = *static_cast<const lights::DirectionalLight*>(hdl);
	if(direction != nullptr)
		*direction = util::pun<Vec3>(light.direction);
	return true;
}

Boolean world_get_dir_light_radiance(LightHdl hdl, Vec3* radiance) {
	CHECK_NULLPTR(hdl, "directional light handle", false);
	const lights::DirectionalLight& light = *static_cast<const lights::DirectionalLight*>(hdl);
	if(radiance != nullptr)
		*radiance = util::pun<Vec3>(light.radiance);
	return true;
}

Boolean world_set_dir_light_direction(LightHdl hdl, Vec3 direction) {
	CHECK_NULLPTR(hdl, "directional light handle", false);
	lights::DirectionalLight& light = *static_cast<lights::DirectionalLight*>(hdl);
	ei::Vec3 actualDirection = ei::normalize(util::pun<ei::Vec3>(direction));
	if(!ei::approx(ei::len(actualDirection), 1.0f)) {
		logError("[", FUNCTION_NAME, "] Directional light direction cannot be a null vector");
		return false;
	}
	light.direction = util::pun<ei::Vec3>(actualDirection);
	return true;
}

Boolean world_set_dir_light_radiance(LightHdl hdl, Vec3 radiance) {
	CHECK_NULLPTR(hdl, "directional light handle", false);
	lights::DirectionalLight& light = *static_cast<lights::DirectionalLight*>(hdl);
	light.radiance = util::pun<ei::Vec3>(radiance);
	return true;
}

Boolean render_enable_renderer(RendererType type) {
	SceneHandle scene = WorldContainer::instance().get_current_scene();
	if(scene == nullptr) {
		logError("[", FUNCTION_NAME, "] Cannot enable renderer before scene hasn't been set");
		return false;
	}
	ei::IVec2 res = WorldContainer::instance().get_current_scenario()->get_resolution();
	s_imageOutput = std::make_unique<renderer::OutputHandler>(res.x, res.y,
															s_outputTargets);
	switch(type) {
		case RendererType::RENDERER_CPU_PT: {
			s_currentRenderer = std::make_unique<renderer::CpuPathTracer>(
				WorldContainer::instance().get_current_scene());
			return true;
		}
		case RendererType::RENDERER_GPU_PT: {
			s_currentRenderer = std::make_unique<renderer::GpuPathTracer>(
				WorldContainer::instance().get_current_scene());
			return true;
		}
		default: {
			logError("[", FUNCTION_NAME, "] Unknown renderer type");
			return false;
		}
	}
}

Boolean render_iterate() {
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] No renderer is currently set");
		return false;
	}
	if(s_imageOutput == nullptr) {
		logError("[", FUNCTION_NAME, "] No rendertarget is currently set");
		return false;
	}
	s_currentRenderer->iterate(*s_imageOutput);
	return true;
}

Boolean render_reset() {
	if (s_currentRenderer != nullptr)
		s_currentRenderer->reset();
	return true;
}

Boolean render_get_screenshot() {
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] No renderer is currently set");
		return false;
	}
	if(s_imageOutput == nullptr) {
		logError("[", FUNCTION_NAME, "] No rendertarget is currently set");
		return false;
	}
	// TODO: this is just for debugging!
	return false;
}

Boolean render_save_screenshot(const char* filename) {
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

	return true;
}

Boolean render_enable_render_target(RenderTarget target, Boolean variance) {
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
	return true;
}

Boolean render_disable_render_target(RenderTarget target, Boolean variance) {
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
	return true;
}

Boolean render_enable_variance_render_targets() {
	for(u32 target : renderer::OutputValue::iterator)
		s_outputTargets.set(target << 8u);
	return true;
}

Boolean render_enable_non_variance_render_targets() {
	for(u32 target : renderer::OutputValue::iterator)
		s_outputTargets.set(target);
	return true;
}

Boolean render_enable_all_render_targets() {
	return render_enable_variance_render_targets()
		&& render_enable_non_variance_render_targets();
}

Boolean render_disable_variance_render_targets() {
	for(u32 target : renderer::OutputValue::iterator)
		s_outputTargets.clear(target << 8u);
	return true;
}

Boolean render_disable_non_variance_render_targets() {
	for(u32 target : renderer::OutputValue::iterator)
		s_outputTargets.clear(target);
	return true;
}

Boolean render_disable_all_render_targets() {
	return render_disable_variance_render_targets()
		&& render_disable_non_variance_render_targets();
}

void CDECL profiling_enable() {
	Profiler::instance().set_enabled(true);
}

void CDECL profiling_disable() {
	Profiler::instance().set_enabled(false);
}

Boolean CDECL profiling_set_level(ProfilingLevel level) {
	switch(level) {
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
	}
	return false;
}

Boolean CDECL profiling_save_current_state(const char* path) {
	CHECK_NULLPTR(path, "file path", false);
	Profiler::instance().save_current_state(path);
	return true;
}

Boolean CDECL profiling_save_snapshots(const char* path) {
	CHECK_NULLPTR(path, "file path", false);
	Profiler::instance().save_snapshots(path);
	return true;
}

const char* CDECL profiling_get_current_state() {
	std::string str = Profiler::instance().save_current_state();
	char* buffer = new char[str.size() + 1u];
	std::memcpy(buffer, str.c_str(), str.size());
	buffer[str.size()] = '\0';
	return buffer;
}

const char* CDECL profiling_get_snapshots() {
	std::string str = Profiler::instance().save_snapshots();
	char* buffer = new char[str.size() + 1u];
	std::memcpy(buffer, str.c_str(), str.size());
	buffer[str.size()] = '\0';
	return buffer;
}

void CDECL profiling_reset() {
	Profiler::instance().reset_all();
}

Boolean mufflon_initialize(void(*logCallback)(const char*, int)) {
	// Only once per process do we register/unregister the message handler
	static bool initialized = false;
	s_logCallback = logCallback;
	if (!initialized) {
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

		if (!gladLoadGL()) {
			logError("[", FUNCTION_NAME, "] gladLoadGL failed");
			return false;
		}

		// Set the CUDA device to initialize the context
		int count = 0;
		cuda::check_error(cudaGetDeviceCount(&count));
		if (count > 0) {
			logInfo("[", FUNCTION_NAME, "] Found ", count, " CUDA-capable "
				"devices; initializing device 0");
			cuda::check_error(cudaSetDevice(0u));
		}

		initialized = true;
	}
	return initialized;
}

CORE_API void CDECL mufflon_destroy() {
	WorldContainer::clear_instance();
	s_imageOutput.reset();
	s_currentRenderer.reset();
}
