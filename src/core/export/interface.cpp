#include "core_interface.h"
#include "plugin/texture_plugin.hpp"
#include "util/log.hpp"
#include "util/byte_io.hpp"
#include "util/parallel.hpp"
#include "util/punning.hpp"
#include "util/degrad.hpp"
#include "util/indexed_string_map.hpp"
#include "ei/vector.hpp"
#include "profiler/cpu_profiler.hpp"
#include "profiler/gpu_profiler.hpp"
#include "core/renderer/renderer.hpp"
#include "core/renderer/renderers.hpp"
#include "core/cameras/pinhole.hpp"
#include "core/cameras/focus.hpp"
#include "core/scene/object.hpp"
#include "core/scene/world_container.hpp"
#include "core/scene/geometry/polygon.hpp"
#include "core/scene/geometry/sphere.hpp"
#include "core/scene/lights/lights.hpp"
#include "core/scene/materials/material.hpp"
#include "core/scene/textures/interface.hpp"
#include "mffloader/interface/mff_interface.h"
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <fstream>
#include <type_traits>
#include <mutex>
#include <fstream>
#include <vector>
#include <set>

#ifdef MUFFLON_ENABLE_OPEN_DENOISE
#include <OpenImageDenoise/oidn.hpp>
#endif // MUFFLON_ENABLE_OPEN_DENOISE

#ifdef _WIN32
#include <windows.h>
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
renderer::IRenderer* s_currentRenderer;
// Current iteration counter
std::unique_ptr<renderer::IOutputHandler> s_imageOutput;
std::unique_ptr<float[]> s_screenTexture;
int s_screenTextureNumChannels;
WorldContainer& s_world = WorldContainer::instance();
static void(*s_logCallback)(const char*, int);
// Holds the CUDA device index
int s_cudaDevIndex = -1;
// Holds the last error for the GUI to display
std::string s_lastError;
// Mutex for exclusive renderer access: during an iteration no other thread may change renderer properties
std::mutex s_iterationMutex{};
std::mutex s_screenTextureMutex{};
// Log file
std::ofstream s_logFile;
// Set of render targets to enable/disable
std::set<std::pair<std::string, bool>> s_targetsToEnable;
std::set<std::pair<std::string, bool>> s_targetsToDisable;

// Plugin container
std::vector<TextureLoaderPlugin> s_plugins;

// List of renderers
util::IndexedStringMap<std::vector<std::unique_ptr<renderer::IRenderer>>> s_renderers;

constexpr VertexAttributeHdl INVALID_POLY_VATTR_HANDLE{ ATTRTYPE_COUNT, nullptr };
constexpr FaceAttributeHdl INVALID_POLY_FATTR_HANDLE{ ATTRTYPE_COUNT, nullptr };
constexpr SphereAttributeHdl INVALID_SPHERE_ATTR_HANDLE{ ATTRTYPE_COUNT, nullptr };

// Initializes all renderers
template < bool initOpenGL, std::size_t I = 0u >
inline void init_renderers() {
	if constexpr(I == 0u && !initOpenGL)
		s_renderers.clear();

	using RendererType = typename renderer::Renderers::Type<I>;
	std::vector<std::unique_ptr<renderer::IRenderer>>* renderers = s_renderers.find(RendererType::get_name_static());

	// Only initialize opengl renderers if requested (because of deferred context init)
	if constexpr(RendererType::may_use_device(Device::OPENGL)) {
		if(initOpenGL) {// deferred init
			if(renderers == nullptr)
				renderers = &s_renderers.get(s_renderers.insert(std::string(RendererType::get_name_static()), {}));
			renderers->push_back(std::make_unique<RendererType>());
		}
	}
	// Only initialize CUDA renderers if CUDA is enabled
	else if(!initOpenGL && (s_cudaDevIndex >= 0 || !RendererType::may_use_device(Device::CUDA))) {
		if(renderers == nullptr)
			renderers = &s_renderers.get(s_renderers.insert(std::string(RendererType::get_name_static()), {}));
		renderers->push_back(std::make_unique<RendererType>());
	}
	if constexpr(I + 1u < renderer::Renderers::size)
		init_renderers<initOpenGL, I + 1u>();
}

// Function delegating the logger output to the applications handle, if applicable
inline void delegateLog(LogSeverity severity, const std::string& message) {
	TRY
	if(s_logCallback != nullptr)
		s_logCallback(message.c_str(), static_cast<int>(severity));
	s_logFile << message << std::endl;
	if(severity == LogSeverity::ERROR || severity == LogSeverity::FATAL_ERROR) {
		s_lastError = message;
	}
	CATCH_ALL(;)
}

inline std::string replace_screenshot_filename_tags(std::string name, const StringView targetName,
													const bool variance) {
	// Replace tags in the file name
	auto replacer = [](std::string str, const StringView from, const StringView to) {
		if(const auto start = str.find(from.data()); start != std::string::npos) {
			return str.replace(start, from.length(), to.data());
		}
		return str;
	};
	// Unfortunately we cannot replace the scene path since we don't know it,
	// so that's still left to the front-end
	name = replacer(name, "#scenario", s_world.get_current_scenario()->get_name());
	name = replacer(name, "#renderer", s_currentRenderer->get_name());
	name = replacer(name, "#shortrenderer", s_currentRenderer->get_short_name());
	name = replacer(name, "#iteration", std::to_string(render_get_current_iteration()));
	name = replacer(name, "#frame", std::to_string(s_world.get_frame_current()));
	if(variance)
		name = replacer(name, "#target", std::string(targetName) + "(Variance)");
	else
		name = replacer(name, "#target", targetName);
	// TODO: enum to string would be nice here...
	const std::string usedDevices = std::string(s_currentRenderer->uses_device(Device::CPU) ? ",CPU" : "")
		+ std::string(s_currentRenderer->uses_device(Device::CUDA) ? ",CUDA" : "")
		+ std::string(s_currentRenderer->uses_device(Device::OPENGL) ? ",OPENGL" : "");
	name = replacer(name, "#devices", usedDevices.size() > 0 ? StringView(&usedDevices[1], usedDevices.size() - 1) : usedDevices);
	name = replacer(name, "#camera", s_world.get_current_scenario()->get_camera()->get_name());
	return name;
}

} // namespace

const char* core_get_dll_error() {
	TRY
	return s_lastError.c_str();
	CATCH_ALL(nullptr)
}

Boolean core_set_log_level(LogLevel level) {
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

Boolean core_set_lod_loader(Boolean(*func)(ObjectHdl, uint32_t)) {
	TRY
	CHECK_NULLPTR(func, "LoD loader function", false);
	s_world.set_lod_loader_function(reinterpret_cast<std::uint32_t(*)(ObjectHandle, u32)>(func));
	return true;
	CATCH_ALL(false)
}

Boolean core_get_target_image(const char* name, Boolean variance, const float** ptr) {
	TRY
	CHECK_NULLPTR(s_currentRenderer, "current renderer", false);
	std::scoped_lock iterLock{ s_iterationMutex };
		
	// If there's no output yet, we "return" a nullptr
	if(s_imageOutput != nullptr) {
		auto data = s_imageOutput->get_data(name, variance);
		std::scoped_lock screenLock{ s_screenTextureMutex };
		s_screenTexture = std::move(data);
		s_screenTextureNumChannels = s_imageOutput->get_num_channels(name);
		if(ptr != nullptr)
			*ptr = reinterpret_cast<const float*>(s_screenTexture.get());
	} else if(ptr != nullptr) {
		*ptr = nullptr;
	}
	return true;
	CATCH_ALL(false)
}

Boolean core_get_target_image_num_channels(int* numChannels) {
	CHECK_NULLPTR(s_screenTexture, "screen texture", false);
	*numChannels = s_screenTextureNumChannels;
	return true;
}

Boolean core_copy_screen_texture_rgba32(Vec4* ptr, const float factor) {
	TRY
	CHECK_NULLPTR(s_currentRenderer, "current renderer", false);
	std::scoped_lock lock{ s_screenTextureMutex };
	if(ptr != nullptr && s_screenTexture != nullptr) {
		const int numPixels = s_imageOutput->get_width() * s_imageOutput->get_height();
		const float* texData = s_screenTexture.get();
#pragma PARALLEL_FOR
		for(int i = 0; i < numPixels; ++i) {
			Vec4 pixel { 0.f, 0.f, 0.f, 0.f };
			float* dst = reinterpret_cast<float*>(&pixel);
			int idx = i * s_screenTextureNumChannels;
			for(int c = 0; c < s_screenTextureNumChannels; ++c)
				dst[c] = factor * texData[idx+c];
			ptr[i] = pixel;
		}
	}
	return true;
	CATCH_ALL(false)
}

Boolean core_get_pixel_info(uint32_t x, uint32_t y, Boolean borderClamp, float* r, float* g, float* b, float* a) {
	TRY
	CHECK(borderClamp || ((int)x < s_imageOutput->get_width() && (int)y < s_imageOutput->get_height()),
		  "Pixel coordinates are out of bounds", false);
	*r = *g = *b = *a = 0.0f;
	if(s_screenTexture) {
		auto coordX = std::min<std::uint32_t>(s_imageOutput->get_width() - 1, x);
		auto coordY = std::min<std::uint32_t>(s_imageOutput->get_height() - 1, y);
		int idx = (coordX + coordY * s_imageOutput->get_width()) * s_screenTextureNumChannels;
		switch(s_screenTextureNumChannels) {
			case 4: *a = s_screenTexture[idx+3]; // FALLTHROUGH
			case 3: *b = s_screenTexture[idx+2]; // FALLTHROUGH
			case 2: *g = s_screenTexture[idx+1]; // FALLTHROUGH
			case 1: *r = s_screenTexture[idx];
		}
	}
	return true;
	CATCH_ALL(false)
}

Boolean polygon_reserve(LodHdl lvlDtl, size_t vertices, size_t edges, size_t tris, size_t quads) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", false);
	static_cast<Lod*>(lvlDtl)->template get_geometry<Polygons>().reserve(vertices, edges, tris, quads);
	return true;
	CATCH_ALL(false)
}

VertexAttributeHdl polygon_request_vertex_attribute(LodHdl lvlDtl, const char* name,
													GeomAttributeType type) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", INVALID_POLY_VATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_POLY_VATTR_HANDLE);
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	auto handle = lod.template get_geometry<Polygons>().add_vertex_attribute(name, static_cast<AttributeType>(type));
	return VertexAttributeHdl{ type, name };
	CATCH_ALL(INVALID_POLY_VATTR_HANDLE)
}

FaceAttributeHdl polygon_request_face_attribute(LodHdl lvlDtl, const char* name,
												GeomAttributeType type) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", INVALID_POLY_FATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_POLY_FATTR_HANDLE);
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	auto handle = lod.template get_geometry<Polygons>().add_face_attribute(name, static_cast<AttributeType>(type));
	return FaceAttributeHdl{ type, name };
	CATCH_ALL(INVALID_POLY_FATTR_HANDLE)
}

VertexHdl polygon_add_vertex(LodHdl lvlDtl, Vec3 point, Vec3 normal, Vec2 uv) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", VertexHdl{ INVALID_INDEX });
	PolyVHdl hdl = static_cast<Lod*>(lvlDtl)->template get_geometry<Polygons>().add(
		util::pun<ei::Vec3>(point), util::pun<ei::Vec3>(normal),
		util::pun<ei::Vec2>(uv));
	if(!hdl.is_valid()) {
		logError("[", FUNCTION_NAME, "] Error adding vertex to polygon");
		return VertexHdl{ INVALID_INDEX };
	}
	return VertexHdl{ static_cast<IndexType>(hdl.idx()) };
	CATCH_ALL(VertexHdl{ INVALID_INDEX })
}

FaceHdl polygon_add_triangle(LodHdl lvlDtl, UVec3 vertices) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", FaceHdl{ INVALID_INDEX });

	PolyFHdl hdl = static_cast<Lod*>(lvlDtl)->template get_geometry<Polygons>().add(
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

FaceHdl polygon_add_triangle_material(LodHdl lvlDtl, UVec3 vertices,
										 MatIdx idx) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", FaceHdl{ INVALID_INDEX });
	PolyFHdl hdl = static_cast<Lod*>(lvlDtl)->template get_geometry<Polygons>().add(
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

FaceHdl polygon_add_quad(LodHdl lvlDtl, UVec4 vertices) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", FaceHdl{ INVALID_INDEX });
	PolyFHdl hdl = static_cast<Lod*>(lvlDtl)->template get_geometry<Polygons>().add(
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

FaceHdl polygon_add_quad_material(LodHdl lvlDtl, UVec4 vertices,
									 MatIdx idx) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", FaceHdl{ INVALID_INDEX });
	PolyFHdl hdl = static_cast<Lod*>(lvlDtl)->template get_geometry<Polygons>().add(
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

VertexHdl polygon_add_vertex_bulk(LodHdl lvlDtl, size_t count, const BulkLoader* points,
								  const BulkLoader* normals, const BulkLoader* uvs,
								  const AABB* aabb, size_t* pointsRead, size_t* normalsRead,
								  size_t* uvsRead) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(points, "points stream descriptor", VertexHdl{ INVALID_INDEX });
	CHECK_NULLPTR(uvs, "UV coordinates stream descriptor", VertexHdl{ INVALID_INDEX });
	Lod& lod = *static_cast<Lod*>(lvlDtl);

	std::unique_ptr<util::IByteReader> pointReader;
	std::unique_ptr<util::IByteReader> normalReader;
	std::unique_ptr<util::IByteReader> uvReader;
	std::unique_ptr<std::istream> pointStream;
	std::unique_ptr<std::istream> normalStream;
	std::unique_ptr<std::istream> uvStream;
	std::unique_ptr<util::ArrayStreamBuffer> pointBuffer;
	std::unique_ptr<util::ArrayStreamBuffer> normalBuffer;
	std::unique_ptr<util::ArrayStreamBuffer> uvBuffer;
	if(points->type == BulkType::BULK_FILE) {
		pointReader = std::make_unique<util::FileReader>(points->descriptor.file);
	} else {
		pointBuffer = std::make_unique<util::ArrayStreamBuffer>(points->descriptor.bytes, count * sizeof(Vec3));
		pointStream = std::make_unique<std::istream>(pointBuffer.get());
		pointReader = std::make_unique<util::StreamReader>(*pointStream);
	}
	if(normals != nullptr && normals->type == BulkType::BULK_FILE) {
		normalReader = std::make_unique<util::FileReader>(normals->descriptor.file);
	} else if(normals != nullptr) {
		normalBuffer = std::make_unique<util::ArrayStreamBuffer>(normals->descriptor.bytes, count * sizeof(Vec3));
		normalStream = std::make_unique<std::istream>(normalBuffer.get());
		normalReader = std::make_unique<util::StreamReader>(*normalStream);
	}
	if(uvs->type == BulkType::BULK_FILE) {
		uvReader = std::make_unique<util::FileReader>(uvs->descriptor.file);
	} else {
		uvBuffer = std::make_unique<util::ArrayStreamBuffer>(uvs->descriptor.bytes, count * sizeof(Vec2));
		uvStream = std::make_unique<std::istream>(uvBuffer.get());
		uvReader = std::make_unique<util::StreamReader>(*uvStream);
	}

	Polygons::VertexBulkReturn info;
	if(aabb != nullptr) {
		if(normalReader != nullptr)
			info = lod.template get_geometry<Polygons>().add_bulk(count, *pointReader, *normalReader,
																  *uvReader, util::pun<ei::Box>(*aabb));
		else
			info = lod.template get_geometry<Polygons>().add_bulk(count, *pointReader,
																  *uvReader, util::pun<ei::Box>(*aabb));
	} else {
		if(normalReader != nullptr)
			info = lod.template get_geometry<Polygons>().add_bulk(count, *pointReader, *normalReader, *uvReader);
		else
			info = lod.template get_geometry<Polygons>().add_bulk(count, *pointReader, *uvReader);
	}

	if(pointsRead != nullptr)
		*pointsRead = info.readPoints;
	if(normalsRead != nullptr)
		*normalsRead = info.readNormals;
	if(uvsRead != nullptr)
		*uvsRead = info.readUvs;
	return VertexHdl{ static_cast<IndexType>(info.handle.idx()) };
	CATCH_ALL(VertexHdl{ INVALID_INDEX })
}

Boolean polygon_set_vertex_attribute(LodHdl lvlDtl, const VertexAttributeHdl attr,
									 VertexHdl vertex, const void* value) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", false);
	CHECK_NULLPTR(attr.name, "attribute name", false);
	CHECK_NULLPTR(value, "attribute value", false);
	CHECK_GEQ_ZERO(vertex, "vertex index", false);
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	if(vertex >= static_cast<int>(lod.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 vertex, " >= ", lod.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return false;
	}

	auto& polys = lod.template get_geometry<Polygons>();
	const auto hdl = polys.find_vertex_attribute(attr.name, static_cast<AttributeType>(attr.type));
	if(!hdl.has_value()) {
		logError("[", FUNCTION_NAME, "] Could not retrieve vertex attribute handle");
		return false;
	}
	const auto elemSize = get_attribute_size(static_cast<AttributeType>(attr.type));
	char* data = polys.template acquire<Device::CPU, char>(hdl.value())
		+ elemSize * vertex;
	std::memcpy(data, value, elemSize);
	lod.template get_geometry<Polygons>().mark_changed(Device::CPU);
	return true;
	CATCH_ALL(false)
}

Boolean polygon_set_vertex_normal(LodHdl lvlDtl, VertexHdl vertex, Vec3 normal) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", false);
	CHECK_GEQ_ZERO(vertex, "vertex index", false);
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	if(vertex >= static_cast<int>(lod.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 vertex, " >= ", lod.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return false;
	}

	auto hdl = lod.get_geometry<Polygons>().get_normals_hdl();
	lod.get_geometry<Polygons>().acquire<Device::CPU, OpenMesh::Vec3f>(hdl)[vertex] = util::pun<OpenMesh::Vec3f>(normal);
	lod.get_geometry<Polygons>().mark_changed(Device::CPU);
	return true;
	CATCH_ALL(false)
}

Boolean polygon_set_vertex_uv(LodHdl lvlDtl, VertexHdl vertex, Vec2 uv) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", false);
	CHECK_GEQ_ZERO(vertex, "vertex index", false);
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	if(vertex >= static_cast<int>(lod.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 vertex, " >= ", lod.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return false;
	}

	auto hdl = lod.get_geometry<Polygons>().get_uvs_hdl();
	lod.get_geometry<Polygons>().acquire<Device::CPU, OpenMesh::Vec2f>(hdl)[vertex] = util::pun<OpenMesh::Vec2f>(uv);
	lod.get_geometry<Polygons>().mark_changed(Device::CPU);
	return true;
	CATCH_ALL(false)
}

Boolean polygon_set_face_attribute(LodHdl lvlDtl, const FaceAttributeHdl attr,
								   FaceHdl face, const void* value) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", false);
	CHECK_NULLPTR(attr.name, "attribute name", false);
	CHECK_NULLPTR(value, "attribute value", false);
	CHECK_GEQ_ZERO(face, "face index", false);

	Lod& lod = *static_cast<Lod*>(lvlDtl);
	if(face >= static_cast<int>(lod.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 face, " >= ", lod.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return false;
	}

	auto& polys = lod.template get_geometry<Polygons>();
	const auto hdl = polys.find_face_attribute(attr.name, static_cast<AttributeType>(attr.type));
	if(!hdl.has_value()) {
		logError("[", FUNCTION_NAME, "] Could not retrieve face attribute handle");
		return false;
	}
	const auto elemSize = get_attribute_size(static_cast<AttributeType>(attr.type));
	char* data = polys.template acquire<Device::CPU, char>(hdl.value())
		+ elemSize * face;
	std::memcpy(data, value, elemSize);
	lod.template get_geometry<Polygons>().mark_changed(Device::CPU);
	return true;
	CATCH_ALL(false)
}

Boolean polygon_set_material_idx(LodHdl lvlDtl, FaceHdl face, MatIdx idx) {
	// TODO: polygons have the invariant that their material never changes.
	// This interface violates this invariant directly, remove?
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", false);
	CHECK_GEQ_ZERO(face, "face index", false);
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	if(face >= static_cast<int>(lod.template get_geometry<Polygons>().get_face_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 face, " >= ", lod.template get_geometry<Polygons>().get_face_count(),
				 ")");
		return false;
	}

	auto hdl = lod.get_geometry<Polygons>().get_material_indices_hdl();
	lod.get_geometry<Polygons>().acquire<Device::CPU, MaterialIndex>(hdl)[face] = idx;
	lod.get_geometry<Polygons>().mark_changed(Device::CPU);
	return true;
	CATCH_ALL(false)
}

size_t polygon_set_vertex_attribute_bulk(LodHdl lvlDtl, const VertexAttributeHdl attr,
										 VertexHdl startVertex, size_t count,
										 const BulkLoader* stream) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", INVALID_SIZE);
	CHECK_NULLPTR(attr.name, "attribute name", INVALID_SIZE);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK_GEQ_ZERO(startVertex, "start vertex index", INVALID_SIZE);
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	if(startVertex >= static_cast<int>(lod.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 startVertex, " >= ", lod.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return INVALID_SIZE;
	}

	const auto elemSize = get_attribute_size(static_cast<AttributeType>(attr.type));
	std::unique_ptr<util::IByteReader> attrReader;
	std::unique_ptr<util::ArrayStreamBuffer> attrBuffer;
	std::unique_ptr<std::istream> attrStream;
	if(stream->type == BulkType::BULK_FILE) {
		attrReader = std::make_unique<util::FileReader>(stream->descriptor.file);
	} else {
		attrBuffer = std::make_unique<util::ArrayStreamBuffer>(stream->descriptor.bytes, count * elemSize);
		attrStream = std::make_unique<std::istream>(attrBuffer.get());
		attrReader = std::make_unique<util::StreamReader>(*attrStream);
	}

	auto& polys = lod.template get_geometry<Polygons>();
	const auto hdl = polys.find_vertex_attribute(attr.name, static_cast<AttributeType>(attr.type));
	if(!hdl.has_value()) {
		logError("[", FUNCTION_NAME, "] Could not retrieve vertex attribute handle");
		return false;
	}
	polys.add_bulk(hdl.value(), PolyVHdl{ static_cast<int>(startVertex) },
				   count, *attrReader);
	return true;
	CATCH_ALL(INVALID_SIZE)
}

size_t polygon_set_face_attribute_bulk(LodHdl lvlDtl, const FaceAttributeHdl attr,
									   FaceHdl startFace, size_t count,
									   const BulkLoader* stream) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", INVALID_SIZE);
	CHECK_NULLPTR(attr.name, "attribute name", INVALID_SIZE);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK_GEQ_ZERO(startFace, "start face index", INVALID_SIZE);
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	if(startFace >= static_cast<int>(lod.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 startFace, " >= ", lod.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return INVALID_SIZE;
	}

	const auto elemSize = get_attribute_size(static_cast<AttributeType>(attr.type));
	std::unique_ptr<util::IByteReader> attrReader;
	std::unique_ptr<util::ArrayStreamBuffer> attrBuffer;
	std::unique_ptr<std::istream> attrStream;
	if(stream->type == BulkType::BULK_FILE) {
		attrReader = std::make_unique<util::FileReader>(stream->descriptor.file);
	} else {
		attrBuffer = std::make_unique<util::ArrayStreamBuffer>(stream->descriptor.bytes, count * elemSize);
		attrStream = std::make_unique<std::istream>(attrBuffer.get());
		attrReader = std::make_unique<util::StreamReader>(*attrStream);
	}

	auto& polys = lod.template get_geometry<Polygons>();
	const auto hdl = polys.find_face_attribute(attr.name, static_cast<AttributeType>(attr.type));
	if(!hdl.has_value()) {
		logError("[", FUNCTION_NAME, "] Could not retrieve vertex attribute handle");
		return false;
	}
	polys.add_bulk(hdl.value(), PolyFHdl{ static_cast<int>(startFace) },
				   count, *attrReader);
	return true;
	CATCH_ALL(INVALID_SIZE)
}

size_t polygon_set_material_idx_bulk(LodHdl lvlDtl, FaceHdl startFace, size_t count,
									 const BulkLoader* stream) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", false);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK_GEQ_ZERO(startFace, "start face index", INVALID_SIZE);
	if(count == 0u)
		return 0u;
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	if(startFace >= static_cast<int>(lod.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 startFace, " >= ", lod.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return INVALID_SIZE;
	}
	std::unique_ptr<util::IByteReader> matReader;
	std::unique_ptr<util::ArrayStreamBuffer> matBuffer;
	std::unique_ptr<std::istream> matStream;
	if(stream->type == BulkType::BULK_FILE) {
		matReader = std::make_unique<util::FileReader>(stream->descriptor.file);
	} else {
		matBuffer = std::make_unique<util::ArrayStreamBuffer>(stream->descriptor.bytes, count * sizeof(MaterialIndex));
		matStream = std::make_unique<std::istream>(matBuffer.get());
		matReader = std::make_unique<util::StreamReader>(*matStream);
	}

	FaceAttributeHandle hdl = lod.template get_geometry<Polygons>().get_material_indices_hdl();
	return lod.template get_geometry<Polygons>().add_bulk(hdl, PolyFHdl{ static_cast<int>(startFace) },
															 count, *matReader);
	CATCH_ALL(INVALID_SIZE)
}

size_t polygon_get_vertex_count(LodHdl lvlDtl) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", INVALID_SIZE);
	const Lod& lod = *static_cast<const Lod*>(lvlDtl);
	return lod.template get_geometry<Polygons>().get_vertex_count();
	CATCH_ALL(INVALID_SIZE)
}

size_t polygon_get_edge_count(LodHdl lvlDtl) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", INVALID_SIZE);
	const Lod& lod = *static_cast<const Lod*>(lvlDtl);
	return lod.template get_geometry<Polygons>().get_edge_count();
	CATCH_ALL(INVALID_SIZE)
}

size_t polygon_get_face_count(LodHdl lvlDtl) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", INVALID_SIZE);
	const Lod& lod = *static_cast<const Lod*>(lvlDtl);
	return lod.template get_geometry<Polygons>().get_face_count();
	CATCH_ALL(INVALID_SIZE)
}

size_t polygon_get_triangle_count(LodHdl lvlDtl) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", INVALID_SIZE);
	const Lod& lod = *static_cast<const Lod*>(lvlDtl);
	return lod.template get_geometry<Polygons>().get_triangle_count();
	CATCH_ALL(INVALID_SIZE)
}

size_t polygon_get_quad_count(LodHdl lvlDtl) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", INVALID_SIZE);
	const Lod& lod = *static_cast<const Lod*>(lvlDtl);
	return lod.template get_geometry<Polygons>().get_quad_count();
	CATCH_ALL(INVALID_SIZE)
}

Boolean polygon_get_bounding_box(LodHdl lvlDtl, Vec3* min, Vec3* max) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", false);
	const Lod& lod = *static_cast<const Lod*>(lvlDtl);
	const ei::Box& aabb = lod.template get_geometry<Polygons>().get_bounding_box();
	if(min != nullptr)
		*min = util::pun<Vec3>(aabb.min);
	if(max != nullptr)
		*max = util::pun<Vec3>(aabb.max);
	return true;
	CATCH_ALL(false)
}

Boolean spheres_reserve(LodHdl lvlDtl, size_t count) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", false);
	static_cast<Lod*>(lvlDtl)->template get_geometry<Spheres>().reserve(count);
	return true;
	CATCH_ALL(false)
}

SphereAttributeHdl spheres_request_attribute(LodHdl lvlDtl, const char* name,
											 GeomAttributeType type) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", INVALID_SPHERE_ATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_SPHERE_ATTR_HANDLE);
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	auto handle = lod.template get_geometry<Spheres>().add_attribute(name, static_cast<AttributeType>(type));
	return SphereAttributeHdl{ type, name };
	CATCH_ALL(INVALID_SPHERE_ATTR_HANDLE)
}

SphereHdl spheres_add_sphere(LodHdl lvlDtl, Vec3 point, float radius) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", SphereHdl{ INVALID_INDEX });
	SphereVHdl hdl = static_cast<Lod*>(lvlDtl)->template get_geometry<Spheres>().add(
		util::pun<ei::Vec3>(point), radius);
	return SphereHdl{ static_cast<IndexType>(hdl) };
	CATCH_ALL(SphereHdl{ INVALID_INDEX })
}

SphereHdl spheres_add_sphere_material(LodHdl lvlDtl, Vec3 point, float radius,
									  MatIdx idx) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", SphereHdl{ INVALID_INDEX });
	SphereVHdl hdl = static_cast<Lod*>(lvlDtl)->template get_geometry<Spheres>().add(
		util::pun<ei::Vec3>(point), radius, idx);
	return SphereHdl{ static_cast<IndexType>(hdl) };
	CATCH_ALL(SphereHdl{ INVALID_INDEX })
}

SphereHdl spheres_add_sphere_bulk(LodHdl lvlDtl, size_t count, const BulkLoader* stream,
								  const AABB* aabb, size_t* readSpheres) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", SphereHdl{ INVALID_INDEX } );
	CHECK_NULLPTR(stream, "sphere stream descriptor", SphereHdl{ INVALID_INDEX });
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	std::unique_ptr<util::IByteReader> sphereReader;
	std::unique_ptr<util::ArrayStreamBuffer> sphereBuffer;
	std::unique_ptr<std::istream> sphereStream;
	if(stream->type == BulkType::BULK_FILE) {
		sphereReader = std::make_unique<util::FileReader>(stream->descriptor.file);
	} else {
		sphereBuffer = std::make_unique<util::ArrayStreamBuffer>(stream->descriptor.bytes, count * sizeof(ei::Sphere));
		sphereStream = std::make_unique<std::istream>(sphereBuffer.get());
		sphereReader = std::make_unique<util::StreamReader>(*sphereStream);
	}

	Spheres::BulkReturn info;
	if(aabb != nullptr)
		info = lod.template get_geometry<Spheres>().add_bulk(count, *sphereReader, util::pun<ei::Box>(*aabb));
	else
		info = lod.template get_geometry<Spheres>().add_bulk(count, *sphereReader);

	if(readSpheres != nullptr)
		*readSpheres = info.readSpheres;
	return SphereHdl{ static_cast<IndexType>(info.handle) };
	CATCH_ALL(SphereHdl{ INVALID_INDEX })
}

Boolean spheres_set_attribute(LodHdl lvlDtl, const SphereAttributeHdl attr,
						   SphereHdl sphere, const void* value) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", false);
	CHECK_NULLPTR(attr.name, "attribute name", false);
	CHECK_NULLPTR(value, "attribute value", false);
	CHECK_GEQ_ZERO(sphere, "sphere index", false);
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	if(sphere >= static_cast<int>(lod.template get_geometry<Spheres>().get_sphere_count())) {
		logError("[", FUNCTION_NAME, "] Sphere index out of bounds (",
				 sphere, " >= ", lod.template get_geometry<Spheres>().get_sphere_count(),
				 ")");
		return false;
	}


	auto& spheres = lod.template get_geometry<Spheres>();
	const auto hdl = spheres.find_attribute(attr.name, static_cast<AttributeType>(attr.type));
	if(!hdl.has_value()) {
		logError("[", FUNCTION_NAME, "] Could not retrieve face attribute handle");
		return false;
	}
	const auto elemSize = get_attribute_size(static_cast<AttributeType>(attr.type));
	char* data = spheres.template acquire<Device::CPU, char>(hdl.value())
		+ elemSize * sphere;
	std::memcpy(data, value, elemSize);
	lod.template get_geometry<Polygons>().mark_changed(Device::CPU);
	return true;
	CATCH_ALL(false)
}

Boolean spheres_set_material_idx(LodHdl lvlDtl, SphereHdl sphere, MatIdx idx) {
	// TODO: polygons have the invariant that their material never changes.
	// This interface violates this invariant directly, remove?
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", false);
	CHECK_GEQ_ZERO(sphere, "sphere index", false);
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	if(sphere >= static_cast<int>(lod.template get_geometry<Spheres>().get_sphere_count())) {
		logError("[", FUNCTION_NAME, "] Sphere index out of bounds (",
				 sphere, " >= ", lod.template get_geometry<Spheres>().get_sphere_count(),
				 ")");
		return false;
	}

	SphereAttributeHandle hdl = lod.template get_geometry<Spheres>().get_material_indices_hdl();
	lod.template get_geometry<Spheres>().acquire<Device::CPU, MaterialIndex>(hdl)[sphere] = idx;
	lod.template get_geometry<Spheres>().mark_changed(Device::CPU);
	return true;
	CATCH_ALL(false)
}

size_t spheres_set_attribute_bulk(LodHdl lvlDtl, const SphereAttributeHdl attr,
								  SphereHdl startSphere, size_t count,
								  const BulkLoader* stream) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", INVALID_SIZE);
	CHECK_NULLPTR(attr.name, "attribute name", INVALID_SIZE);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK_GEQ_ZERO(startSphere, "start sphere index", INVALID_SIZE);
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	if(startSphere >= static_cast<int>(lod.template get_geometry<Spheres>().get_sphere_count())) {
		logError("[", FUNCTION_NAME, "] Sphere index out of bounds (",
				 startSphere, " >= ", lod.template get_geometry<Spheres>().get_sphere_count(),
				 ")");
		return INVALID_SIZE;
	}
	const auto elemSize = get_attribute_size(static_cast<AttributeType>(attr.type));
	std::unique_ptr<util::IByteReader> attrReader;
	std::unique_ptr<util::ArrayStreamBuffer> attrBuffer;
	std::unique_ptr<std::istream> attrStream;
	if(stream->type == BulkType::BULK_FILE) {
		attrReader = std::make_unique<util::FileReader>(stream->descriptor.file);
	} else {
		attrBuffer = std::make_unique<util::ArrayStreamBuffer>(stream->descriptor.bytes, count * elemSize);
		attrStream = std::make_unique<std::istream>(attrBuffer.get());
		attrReader = std::make_unique<util::StreamReader>(*attrStream);
	}

	auto& spheres = lod.template get_geometry<Spheres>();
	const auto hdl = spheres.find_attribute(attr.name, static_cast<AttributeType>(attr.type));
	if(!hdl.has_value()) {
		logError("[", FUNCTION_NAME, "] Could not retrieve sphere attribute handle");
		return false;
	}
	spheres.add_bulk(hdl.value(), SphereHdl{ static_cast<int>(startSphere) },
				   count, *attrReader);
	return true;
	CATCH_ALL(INVALID_SIZE)
}

size_t spheres_set_material_idx_bulk(LodHdl lvlDtl, SphereHdl startSphere, size_t count,
									 const BulkLoader* stream) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", INVALID_SIZE);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK_GEQ_ZERO(startSphere, "start sphere index", INVALID_SIZE);
	Lod& lod = *static_cast<Lod*>(lvlDtl);
	if(startSphere >= static_cast<int>(lod.template get_geometry<Spheres>().get_sphere_count())) {
		logError("[", FUNCTION_NAME, "] Sphere index out of bounds (",
				 startSphere, " >= ", lod.template get_geometry<Spheres>().get_sphere_count(),
				 ")");
		return INVALID_SIZE;
	}
	std::unique_ptr<util::IByteReader> matReader;
	std::unique_ptr<util::ArrayStreamBuffer> matBuffer;
	std::unique_ptr<std::istream> matStream;
	if(stream->type == BulkType::BULK_FILE) {
		matReader = std::make_unique<util::FileReader>(stream->descriptor.file);
	} else {
		matBuffer = std::make_unique<util::ArrayStreamBuffer>(stream->descriptor.bytes, count * sizeof(MaterialIndex));
		matStream = std::make_unique<std::istream>(matBuffer.get());
		matReader = std::make_unique<util::StreamReader>(*matStream);
	}

	SphereAttributeHandle hdl = lod.template get_geometry<Spheres>().get_material_indices_hdl();
	return lod.template get_geometry<Spheres>().add_bulk(hdl,
															SphereVHdl{ static_cast<size_t>(startSphere) },
															count, *matReader);
	return INVALID_SIZE;
	CATCH_ALL(INVALID_SIZE)
}

size_t spheres_get_sphere_count(LodHdl lvlDtl) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", INVALID_SIZE);
	const Lod& lod = *static_cast<const Lod*>(lvlDtl);
	return lod.template get_geometry<Spheres>().get_sphere_count();
	CATCH_ALL(INVALID_SIZE)
}

Boolean spheres_get_bounding_box(LodHdl lvlDtl, Vec3* min, Vec3* max) {
	TRY
	CHECK_NULLPTR(lvlDtl, "LoD handle", false);
	const Lod& lod = *static_cast<const Lod*>(lvlDtl);
	const ei::Box& aabb = lod.template get_geometry<Spheres>().get_bounding_box();
	if(min != nullptr)
		*min = util::pun<Vec3>(aabb.min);
	if(max != nullptr)
		*max = util::pun<Vec3>(aabb.max);
	return true;
	CATCH_ALL(false)
}

Boolean object_has_lod(ConstObjectHdl obj, LodLevel level) {
	TRY
	CHECK_NULLPTR(obj, "object handle", false);
	return static_cast<const Object*>(obj)->has_lod_available(level);
	CATCH_ALL(false)
}

LodHdl object_add_lod(ObjectHdl obj, LodLevel level) {
	TRY
	CHECK_NULLPTR(obj, "object handle", nullptr);
	Object& object = *static_cast<Object*>(obj);
	return &object.add_lod(level);
	CATCH_ALL(nullptr)
}

Boolean object_get_id(ObjectHdl obj, uint32_t* id) {
	TRY
	CHECK_NULLPTR(obj, "object handle", false);
	const Object& object = *static_cast<const Object*>(obj);
	if(id != nullptr)
		*id = object.get_object_id();
	return true;
	CATCH_ALL(false)
}

Boolean instance_set_transformation_matrix(InstanceHdl inst, const Mat3x4* mat,
										   const Boolean isWorldToInst) {
	TRY
	CHECK_NULLPTR(inst, "instance handle", false);
	CHECK_NULLPTR(mat, "transformation matrix", false);
	ConstInstanceHandle instance = static_cast<ConstInstanceHandle>(inst);
	if(isWorldToInst)
		s_world.set_world_to_instance_transformation(instance, util::pun<ei::Mat3x4>(*mat));
	else
		s_world.set_instance_to_world_transformation(instance, util::pun<ei::Mat3x4>(*mat));
	return true;
	CATCH_ALL(false)
}

Boolean instance_get_transformation_matrix(InstanceHdl inst, Mat3x4* mat) {
	TRY
	CHECK_NULLPTR(inst, "instance handle", false);
	ConstInstanceHandle instance = static_cast<ConstInstanceHandle>(inst);
	if(mat != nullptr)
		*mat = util::pun<Mat3x4>(s_world.compute_instance_to_world_transformation(instance));
	return true;
	CATCH_ALL(false)
}

Boolean instance_get_bounding_box(InstanceHdl inst, Vec3* min, Vec3* max, LodLevel lod) {
	TRY
	CHECK_NULLPTR(inst, "instance handle", false);
	const Instance& instance = *static_cast<ConstInstanceHandle>(inst);
	const ei::Box& aabb = instance.get_bounding_box(lod, s_world.compute_instance_to_world_transformation(&instance));
	if(min != nullptr)
		*min = util::pun<Vec3>(aabb.min);
	if(max != nullptr)
		*max = util::pun<Vec3>(aabb.max);
	return true;
	CATCH_ALL(false)
}

void world_clear_all() {
	TRY
	auto iterLock = std::scoped_lock(s_iterationMutex);
	auto screenLock = std::scoped_lock(s_screenTextureMutex);
	for(std::size_t i = 0u; i < s_renderers.size(); ++i)
		for(auto& renderer : s_renderers.get(i))
			renderer->on_world_clearing();
	WorldContainer::clear_instance();
	s_screenTexture.reset();
	CATCH_ALL(;)
}

void world_reserve_objects_instances(const uint32_t objects, const uint32_t instances) {
	TRY
	s_world.reserve(objects, instances);
	CATCH_ALL(;)
}

void world_reserve_scenarios(const uint32_t scenarios) {
	TRY
	s_world.reserve(scenarios);
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

const char* world_get_object_name(ObjectHdl obj) {
	TRY
	CHECK_NULLPTR(obj, "object handle", nullptr);
	return &reinterpret_cast<mufflon::scene::ConstObjectHandle>(obj)->get_name()[0];
	CATCH_ALL(nullptr)
}

InstanceHdl world_create_instance(ObjectHdl obj, const uint32_t animationFrame) {
	TRY
	CHECK_NULLPTR(obj, "object handle", nullptr);
	ObjectHandle hdl = static_cast<Object*>(obj);
	return static_cast<InstanceHdl>(s_world.create_instance(hdl, animationFrame));
	CATCH_ALL(nullptr)
}

Boolean world_apply_instance_transformation(InstanceHdl inst) {
	TRY
	CHECK_NULLPTR(inst, "instance handle", false);
	s_world.apply_transformation(static_cast<Instance*>(inst));
	return true;
	CATCH_ALL(false)
}

uint32_t world_get_instance_count(const uint32_t frame) {
	TRY
	return static_cast<uint32_t>(s_world.get_instance_count(frame));
	CATCH_ALL(0u)
}

uint32_t world_get_highest_instance_frame() {
	TRY
	return static_cast<uint32_t>(s_world.get_highest_instance_frame());
	CATCH_ALL(0u)
}

InstanceHdl world_get_instance_by_index(uint32_t index, const uint32_t animationFrame)
{
	TRY
	const uint32_t MAX_INDEX = static_cast<uint32_t>(s_world.get_instance_count(animationFrame));
	if (index >= MAX_INDEX) {
		logError("[", FUNCTION_NAME, "] Instance index '", index, "' out of bounds (",
			MAX_INDEX, ')');
		return nullptr;
	}
	return s_world.get_instance(index, animationFrame);
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
	StringView nameView{ name };
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
materials::ShadowingModel convertShadowingModel(ShadowingModel shadowingModel) {
	materials::ShadowingModel shadowingModelNew = materials::ShadowingModel::VCAVITY;
	if(shadowingModel == SHADOWING_SMITH)
		shadowingModelNew = materials::ShadowingModel::SMITH;
	return shadowingModelNew;
}
materials::NDF convertNdf(NormalDistFunction ndf) {
	materials::NDF ndfNew = materials::NDF::GGX;
	if(ndf == NDF_BECKMANN) ndfNew = materials::NDF::BECKMANN;
	if(ndf == NDF_COSINE) logWarning("[", FUNCTION_NAME, "] NDF 'cosine' not supported yet (using ggx).");
	return ndfNew;
}
std::tuple<TextureHandle> to_ctor_args(const LambertParams& params) {
	return {static_cast<TextureHandle>(params.albedo)};
}
std::tuple<TextureHandle, float> to_ctor_args(const OrennayarParams& params) {
	return {static_cast<TextureHandle>(params.albedo), params.roughness};
}
std::tuple<TextureHandle, Spectrum> to_ctor_args(const EmissiveParams& params) {
	return {static_cast<TextureHandle>(params.radiance),
			util::pun<Spectrum>(params.scale)};
}
std::tuple<TextureHandle, TextureHandle, materials::ShadowingModel, materials::NDF> to_ctor_args(const TorranceParams& params) {
	return {static_cast<TextureHandle>(params.albedo),
			static_cast<TextureHandle>(params.roughness),
			convertShadowingModel(params.shadowingModel),
			convertNdf(params.ndf)};
}
std::tuple<Spectrum, float, TextureHandle, materials::ShadowingModel, materials::NDF> to_ctor_args(const WalterParams& params) {
	return {util::pun<Spectrum>(params.absorption),
			params.refractionIndex,
			static_cast<TextureHandle>(params.roughness),
			convertShadowingModel(params.shadowingModel),
			convertNdf(params.ndf)};
}
std::unique_ptr<materials::IMaterial> convert_material(const char* name, const MaterialParams* mat) {
	using namespace materials;
	using std::get;
	CHECK_NULLPTR(name, "material name", nullptr);
	CHECK_NULLPTR(mat, "material parameters", nullptr);

	std::unique_ptr<IMaterial> newMaterial;
	switch(mat->innerType) {
		case MATERIAL_LAMBERT: {
			auto p = to_ctor_args(mat->inner.lambert);
			newMaterial = std::make_unique<Material<Materials::LAMBERT>>( get<0>(p) );
		}	break;
		case MATERIAL_TORRANCE: {
			auto p = to_ctor_args(mat->inner.torrance);
			newMaterial = std::make_unique<Material<Materials::TORRANCE>>( get<0>(p), get<1>(p), get<2>(p), get<3>(p));
		}	break;
		case MATERIAL_WALTER: {
			auto p = to_ctor_args(mat->inner.walter);
			newMaterial = std::make_unique<Material<Materials::WALTER>>(
				get<0>(p), get<1>(p), get<2>(p), get<3>(p), get<4>(p) );
		}	break;
		case MATERIAL_MICROFACET: {
			auto p = to_ctor_args(mat->inner.walter);	// Uses same parametrization as Walter
			newMaterial = std::make_unique<Material<Materials::MICROFACET>>(
				get<0>(p), get<1>(p), get<2>(p), get<3>(p), get<4>(p));
		}	break;
		case MATERIAL_EMISSIVE: {
			auto p = to_ctor_args(mat->inner.emissive);
			newMaterial = std::make_unique<Material<Materials::EMISSIVE>>( get<0>(p), get<1>(p) );
		}	break;
		case MATERIAL_ORENNAYAR: {
			auto p = to_ctor_args(mat->inner.orennayar);
			newMaterial = std::make_unique<Material<Materials::ORENNAYAR>>( get<0>(p), get<1>(p) );
		}	break;
		case MATERIAL_BLEND: {
			// Order materials to reduce the number of cases
			const auto* layerA = &mat->inner.blend.a;
			const auto* layerB = &mat->inner.blend.b;
			if(mat->inner.blend.a.mat->innerType < mat->inner.blend.b.mat->innerType)
				std::swap(layerA, layerB);
			if(layerA->mat->innerType == MATERIAL_LAMBERT && layerB->mat->innerType == MATERIAL_EMISSIVE) {
				newMaterial = std::make_unique<Material<Materials::LAMBERT_EMISSIVE>>(
					layerA->factor, layerB->factor,
					to_ctor_args(layerA->mat->inner.lambert),
					to_ctor_args(layerB->mat->inner.emissive));
			} else if(layerA->mat->innerType == MATERIAL_TORRANCE && layerB->mat->innerType == MATERIAL_LAMBERT) {
				newMaterial = std::make_unique<Material<Materials::TORRANCE_LAMBERT>>(
					layerA->factor, layerB->factor,
					to_ctor_args(layerA->mat->inner.torrance),
					to_ctor_args(layerB->mat->inner.lambert));
			} else if(layerA->mat->innerType == MATERIAL_WALTER && layerB->mat->innerType == MATERIAL_TORRANCE) {
				newMaterial = std::make_unique<Material<Materials::WALTER_TORRANCE>>(
					layerA->factor, layerB->factor,
					to_ctor_args(layerA->mat->inner.walter),
					to_ctor_args(layerB->mat->inner.torrance));
			} else {
				logWarning("[", FUNCTION_NAME, "] Unsupported 'blend' material. The combination of layers is not supported.");
				return nullptr;
			}
		}	break;
		case MATERIAL_FRESNEL: {
			// Order materials to reduce the number of cases
			const auto* layerA = mat->inner.fresnel.a;
			const auto* layerB = mat->inner.fresnel.b;
			if(layerA->innerType == MATERIAL_TORRANCE && layerB->innerType == MATERIAL_LAMBERT) {
				newMaterial = std::make_unique<Material<Materials::FRESNEL_TORRANCE_LAMBERT>>(
					util::pun<ei::Vec2>(mat->inner.fresnel.refractionIndex),
					to_ctor_args(layerA->inner.torrance),
					to_ctor_args(layerB->inner.lambert));
			} else if(layerA->innerType == MATERIAL_TORRANCE && layerB->innerType == MATERIAL_WALTER) {
				newMaterial = std::make_unique<Material<Materials::FRESNEL_TORRANCE_WALTER>>(
					util::pun<ei::Vec2>(mat->inner.fresnel.refractionIndex),
					to_ctor_args(layerA->inner.torrance),
					to_ctor_args(layerB->inner.walter));
			} else {
				logWarning("[", FUNCTION_NAME, "] Unsupported 'fresnel' material. The combination of layers is not supported.");
				return nullptr;
			}
		}	break;
		default:
			logWarning("[", FUNCTION_NAME, "] Unknown material type");
	}

	// Set common properties and add to scene
	newMaterial->set_name(name);
	materials::Medium outerMedium {
		util::pun<ei::Vec2>(mat->outerMedium.refractionIndex),
		util::pun<Spectrum>(mat->outerMedium.absorption)
	};
	newMaterial->set_outer_medium( s_world.add_medium(outerMedium) );
	newMaterial->set_inner_medium( s_world.add_medium(newMaterial->compute_medium(outerMedium)) );
	if(mat->alpha != nullptr)
		newMaterial->set_alpha_texture(static_cast<TextureHandle>(mat->alpha));
	if(mat->displacement.map != nullptr) {
		newMaterial->set_displacement(static_cast<TextureHandle>(mat->displacement.map),
									  static_cast<TextureHandle>(mat->displacement.maxMips),
									  mat->displacement.scale, mat->displacement.bias);
	}

	return newMaterial;
}

// Callback function for OpenGL debug context
void APIENTRY opengl_callback(GLenum /*source*/, GLenum /*type*/, GLuint id,
							  GLenum severity, GLsizei /*length*/,
							  const GLchar* message, const void* /*userParam*/) {
	switch(severity) {
		case GL_DEBUG_SEVERITY_HIGH: logError(message); break;
		case GL_DEBUG_SEVERITY_MEDIUM:
			if (id == 131186) break; // buffer moved from video memory to host
			if (id == 131154) break; // copy buffer during 3d rendering
			logWarning(message); break;
		case GL_DEBUG_SEVERITY_LOW: logInfo(message); break;
		default: logPedantic(message); break;
	}
}

} // namespace ::

MaterialHdl world_add_material(const char* name, const MaterialParams* mat) {
	TRY
	if(mat->innerType >= MATERIAL_NUM) {
		logError("[world_add_material] Invalid material params: type unknown.");
		return nullptr;
	}
	MaterialHandle hdl = s_world.add_material(convert_material(name, mat));

	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Error creating material '",
				 name, "'");
		return nullptr;
	}

	return static_cast<MaterialHdl>(hdl);
	CATCH_ALL(nullptr)
}

IndexType world_get_material_count() {
	return static_cast<IndexType>(s_world.get_material_count());
}

MaterialHdl world_get_material(IndexType index) {
	TRY
	return s_world.get_material(index);
	CATCH_ALL(nullptr)
}

size_t world_get_material_size(MaterialHdl material) {
	TRY
	CHECK_NULLPTR(material, "material handle", 0);
	MaterialHandle hdl = static_cast<MaterialHandle>(material);
	switch(hdl->get_type()) {
		case materials::Materials::EMISSIVE: [[fallthrough]];
		case materials::Materials::LAMBERT: [[fallthrough]];
		case materials::Materials::ORENNAYAR: [[fallthrough]];
		case materials::Materials::TORRANCE: [[fallthrough]];
		case materials::Materials::WALTER:
			return sizeof(MaterialParamsStruct);
		case materials::Materials::LAMBERT_EMISSIVE:
			return sizeof(MaterialParamsStruct) * 3;
		default:
			logWarning("[", FUNCTION_NAME, "] Unknown material type");
			return 0;
	}
	CATCH_ALL(0)
}

const char* world_get_material_name(MaterialHdl material) {
	TRY
	CHECK_NULLPTR(material, "material handle", nullptr);
	MaterialHandle hdl = static_cast<MaterialHandle>(material);
	return hdl->get_name().c_str();
	CATCH_ALL(nullptr)
}

int _world_get_material_data(MaterialHdl material, MaterialParams* buffer) {
	using namespace materials;
	CHECK_NULLPTR(material, "material handle", 0);
	CHECK_NULLPTR(buffer, "material buffer", 0);
	MaterialHandle hdl = static_cast<MaterialHandle>(material);
	const materials::Medium& medium = s_world.get_medium(hdl->get_outer_medium());
	buffer->outerMedium.absorption = util::pun<Vec3>(medium.get_absorption_coeff());
	buffer->outerMedium.refractionIndex = util::pun<Vec2>(medium.get_refraction_index());
	switch(hdl->get_type()) {
		case Materials::EMISSIVE:
			buffer->innerType = MATERIAL_EMISSIVE;
			// TODO
			break;
		case Materials::LAMBERT:
			buffer->innerType = MATERIAL_LAMBERT;
			//TODO
			//buffer->inner.lambert.albedo = static_cast<Material<Materials::LAMBERT>*>(hdl)->get_albedo();
			break;
		case Materials::ORENNAYAR://TODO
			break;
		case Materials::TORRANCE://TODO
			break;
		case Materials::WALTER://TODO
			break;
		case Materials::LAMBERT_EMISSIVE: {
			/*buffer->innerType = MATERIAL_BLEND;
			buffer->inner.blend.a.factor = static_cast<materials::Blend*>(hdl)->get_factor_a();
			buffer->inner.blend.a.mat = buffer + 1;
			int count = _world_get_material_data(MaterialHdl(static_cast<materials::Blend*>(hdl)->get_layer_a()), buffer->inner.blend.a.mat);
			buffer->inner.blend.b.factor = static_cast<materials::Blend*>(hdl)->get_factor_b();
			buffer->inner.blend.b.mat = buffer + 1 + count;
			count += _world_get_material_data(MaterialHdl(static_cast<materials::Blend*>(hdl)->get_layer_b()), buffer->inner.blend.b.mat);
			return count + 1;*/
			//TODO
			break;
		}
		default:
			logWarning("[", FUNCTION_NAME, "] Unknown material type");
			return false;
	}
	return 1;
}

Boolean world_get_material_data(MaterialHdl material, MaterialParams* buffer) {
	TRY
	return _world_get_material_data(material, buffer) >= 1;
	CATCH_ALL(false)
}


CameraHdl world_add_pinhole_camera(const char* name, const Vec3* position, const Vec3* dir,
								   const Vec3* up, const std::uint32_t pathCount, float near,
								   float far, float vFov) {
	TRY
	CHECK_NULLPTR(name, "camera name", nullptr);
	CameraHandle hdl = s_world.add_camera(name,
		std::make_unique<cameras::Pinhole>(
			reinterpret_cast<const ei::Vec3*>(position), reinterpret_cast<const ei::Vec3*>(dir),
			reinterpret_cast<const ei::Vec3*>(up), pathCount, vFov, near, far
	));
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Error creating pinhole camera");
		return nullptr;
	}
	return static_cast<CameraHdl>(hdl);
	CATCH_ALL(nullptr)
}

CameraHdl world_add_focus_camera(const char* name, const Vec3* position, const Vec3* dir,
								 const Vec3* up, const std::uint32_t pathCount, float near, float far,
								 float focalLength, float focusDistance,
								 float lensRad, float chipHeight) {
	TRY
	CHECK_NULLPTR(name, "camera name", nullptr);
	CameraHandle hdl = s_world.add_camera(name,
		std::make_unique<cameras::Focus>(
			reinterpret_cast<const ei::Vec3*>(position), reinterpret_cast<const ei::Vec3*>(dir),
			reinterpret_cast<const ei::Vec3*>(up), pathCount, focalLength, focusDistance,
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

LightHdl world_add_light(const char* name, LightType type, const uint32_t count) {
	TRY
	CHECK_NULLPTR(name, "light name", (LightHdl{7, 0}));
	std::optional<u32> hdl;
	switch(type) {
		case LIGHT_POINT: {
			hdl = s_world.add_light(name, lights::PointLight{}, count);
		} break;
		case LIGHT_SPOT: {
			hdl = s_world.add_light(name, lights::SpotLight{}, count);
		} break;
		case LIGHT_DIRECTIONAL: {
			hdl = s_world.add_light(name, lights::DirectionalLight{}, count);
		} break;
		case LIGHT_ENVMAP:
			logError("[", FUNCTION_NAME, "] This function is not suited for adding background lights; "
					 "please use 'world_add_background_light' instead");
			return LightHdl{ u32(type), hdl.value() };
		default:
			logError("[", FUNCTION_NAME, "] Unknown light type");
			return LightHdl{ 7, 0 };
	}
	if(!hdl.has_value()) {
		logError("[", FUNCTION_NAME, "] Error adding a light");
		return LightHdl{7, 0};
	}
	return LightHdl{u32(type), hdl.value()};
	CATCH_ALL((LightHdl{7, 0}))
}

LightHdl world_add_background_light(const char* name, BackgroundType type) {
	TRY
	CHECK_NULLPTR(name, "light name", (LightHdl{ 7, 0 }));
	std::optional<u32> hdl;
	switch(type) {
		case BackgroundType::BACKGROUND_MONOCHROME: {
			hdl = s_world.add_light(name, lights::BackgroundType::COLORED);
		}	break;
		case BackgroundType::BACKGROUND_ENVMAP: {
			hdl = s_world.add_light(name, lights::BackgroundType::ENVMAP);
		}	break;
		case BackgroundType::BACKGROUND_SKY_HOSEK: {
			hdl = s_world.add_light(name, lights::BackgroundType::SKY_HOSEK);
		}	break;
		default:
			logError("[", FUNCTION_NAME, "] Unknown background type");
			return LightHdl{ 7, 0 };
	}
	if(!hdl.has_value()) {
		logError("[", FUNCTION_NAME, "] Error adding a background light");
		return LightHdl{ 7, 0 };
	}
	return LightHdl{ (u32)LightType::LIGHT_ENVMAP, hdl.value() };
	CATCH_ALL((LightHdl{ 7, 0 }))
}

Boolean world_set_light_name(LightHdl hdl, const char* newName) {
	TRY
	s_world.set_light_name(hdl.index, static_cast<lights::LightType>(hdl.type), newName);
	return true;
	CATCH_ALL(false)
}

Boolean world_remove_light(LightHdl hdl) {
	TRY
	s_world.remove_light(hdl.index, static_cast<lights::LightType>(hdl.type));
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

CameraHdl world_get_camera_by_index(size_t index) {
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
	// Lessen the count by one due to the default background
	return s_world.get_env_light_count() - 1u;
	CATCH_ALL(0u)
}

LightHdl world_get_light_handle(size_t index, LightType type) {
	// Background indices start at 1 because we have a default background light
	if(type == LightType::LIGHT_ENVMAP)
		return LightHdl{ u32(type), u32(index + 1u) };
	return LightHdl{u32(type), u32(index)};
}

LightType world_get_light_type(LightHdl hdl) {
	return static_cast<LightType>(hdl.type);
}

BackgroundType world_get_env_light_type(LightHdl hdl) {
	TRY
	if(hdl.type == (u32)LightType::LIGHT_ENVMAP) {
		// We also have to check if we actually have an envmap or a different background
		const auto* background = s_world.get_background((u32)hdl.index);
		CHECK(background, "background handle", BackgroundType::BACKGROUND_COUNT);
		switch(background->get_type()) {
			case lights::BackgroundType::COLORED:
				return BackgroundType::BACKGROUND_MONOCHROME;
			case lights::BackgroundType::ENVMAP:
				return BackgroundType::BACKGROUND_ENVMAP;
			case lights::BackgroundType::SKY_HOSEK:
				return BackgroundType::BACKGROUND_SKY_HOSEK;
			default:
				logError("[", FUNCTION_NAME, "] Unknown background type");
				return BackgroundType::BACKGROUND_COUNT;
		}
	} else {
		logError("[", FUNCTION_NAME, "] Background type is only available for background lights");
		return BackgroundType::BACKGROUND_COUNT;
	}
	CATCH_ALL(BackgroundType::BACKGROUND_COUNT)
}

const char* world_get_light_name(LightHdl hdl) {
	TRY
	constexpr lights::LightType TYPES[] = {
		lights::LightType::POINT_LIGHT,
		lights::LightType::SPOT_LIGHT,
		lights::LightType::DIRECTIONAL_LIGHT,
		lights::LightType::ENVMAP_LIGHT
	};
	return s_world.get_light_name(hdl.index, TYPES[hdl.type]).data();
	CATCH_ALL(nullptr)
}

SceneHdl world_load_scenario(ScenarioHdl scenario) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	auto lock = std::scoped_lock(s_iterationMutex);
	auto screenLock = std::scoped_lock(s_screenTextureMutex);
	SceneHandle hdl = s_world.load_scene(static_cast<ScenarioHandle>(scenario), s_currentRenderer);
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Failed to load scenario");
		return nullptr;
	}
	ei::IVec2 res = static_cast<ConstScenarioHandle>(scenario)->get_resolution();
	if(s_currentRenderer != nullptr) {
		s_currentRenderer->load_scene(hdl);
		s_imageOutput->resize(res.x, res.y);
	}
	s_screenTexture = nullptr;
	return static_cast<SceneHdl>(hdl);
	CATCH_ALL(nullptr)
}

SceneHdl world_get_current_scene() {
	TRY
	return static_cast<SceneHdl>(s_world.get_current_scene());
	CATCH_ALL(nullptr)
}

Boolean world_finalize(const char** msg) {
	TRY
	switch(s_world.finalize_world()) {
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
	CHECK_NULLPTR(path, "texture path", nullptr);
	auto hdl = s_world.find_texture(path);
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Could not find texture ",
				 path);
		return nullptr;
	}
	return static_cast<TextureHdl>(hdl);
	CATCH_ALL(nullptr)
}

TextureHdl world_add_texture(const char* path, TextureSampling sampling, MipmapType type,
							 TextureCallback callback, void* userParams) {

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
	auto texture = std::make_unique<textures::Texture>(path, texData.width, texData.height,
													   texData.layers, static_cast<textures::MipmapType>(type),
													   static_cast<textures::Format>(texData.format),
													   static_cast<textures::SamplingMode>(sampling),
													   texData.sRgb, false, std::unique_ptr<u8[]>(texData.data));
	if(callback != nullptr) {
		auto cpuTex = texture->template acquire<Device::CPU>();
		for(uint32_t layer = 0u; layer < texData.layers; ++layer) {
			for(uint32_t y = 0u; y < texData.height; ++y) {
				for(uint32_t x = 0u; x < texData.width; ++x) {
					const Pixel texel{ x, y };
					const auto oldVal = util::pun<Vec4>(cpuTex->read(texel, layer));
					const auto newVal = util::pun<ei::Vec4>(callback(x, y, layer, texData.format,
																	 oldVal, userParams));
					cpuTex->write(newVal, texel, layer);
				}
			}
		}
	}

	hdl = s_world.add_texture(std::move(texture));
	return static_cast<TextureHdl>(hdl);
	CATCH_ALL(nullptr)
}

TextureHdl world_add_texture_converted(const char* path, TextureSampling sampling, TextureFormat targetFormat,
									   MipmapType type, TextureCallback callback, void* userParams) {
	TRY
	CHECK_NULLPTR(path, "texture path", nullptr);

	// Give the texture a special name to avoid conflicts with regularly loaded textures
	std::string textureName = std::string(path) + std::string("##CONVERTED_TO_")
		+ std::string(textures::FORMAT_NAME(static_cast<textures::Format>(targetFormat)))
		+ std::string("##");

	// Check if the texture is already loaded
	auto hdl = s_world.find_texture(textureName);
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

	// Create a texture which will serve as a copy mechanism
	textures::CpuTexture tempTex(texData.width, texData.height, texData.layers,
								 static_cast<textures::Format>(texData.format),
								 static_cast<textures::SamplingMode>(sampling),
								 textures::MipmapType::NONE, texData.sRgb, false,
								 std::unique_ptr<u8[]>(texData.data));
	// Create an empty texture that we'll fill with the desired format
	std::unique_ptr<u8[]> texBuffer = nullptr;
	if(type != MipmapType::MIPMAP_NONE)
		texBuffer = std::make_unique<u8[]>(texData.width * texData.height * texData.layers * textures::PIXEL_SIZE(tempTex.get_format())
										   * (1 + ei::ilog2(std::max(texData.width, texData.height))));
	auto finalTex = std::make_unique<textures::Texture>(std::move(textureName), texData.width, texData.height, texData.layers,
														static_cast<textures::MipmapType>(type), static_cast<textures::Format>(targetFormat),
														static_cast<textures::SamplingMode>(sampling),
														texData.sRgb, true, std::move(texBuffer));
	auto cpuFinalTex = finalTex->template acquire<Device::CPU>();
	for(u32 layer = 0u; layer < texData.layers; ++layer) {
		for(u32 y = 0u; y < texData.height; ++y) {
			for(u32 x = 0u; x < texData.width; ++x) {
				const Pixel texel{ x, y };
				cpuFinalTex->write(tempTex.read(texel, layer), texel, layer);
				if(callback != nullptr) {
					const auto oldVal = util::pun<Vec4>(cpuFinalTex->read(texel, layer));
					const auto newVal = util::pun<ei::Vec4>(callback(x, y, layer, texData.format,
																	 oldVal, userParams));
					cpuFinalTex->write(newVal, texel, layer);
				}
			}
		}
	}
	cpuFinalTex->recompute_mipmaps(static_cast<textures::MipmapType>(type));
	finalTex->mark_changed(Device::CPU);

	// The texture will take ownership of the pointer
	hdl = s_world.add_texture(std::move(finalTex));
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
	auto texture = std::make_unique<textures::Texture>(name, 1, 1, 1, textures::MipmapType::NONE, format,
													   static_cast<textures::SamplingMode>(sampling),
													   false, true, move(data));
	hdl = s_world.add_texture(std::move(texture));
	return static_cast<TextureHdl>(hdl);
	CATCH_ALL(nullptr)
}

Boolean world_add_displacement_map(const char* path, TextureHdl* hdlTex, TextureHdl* hdlMips) {
	TRY
	CHECK_NULLPTR(path, "texture path", false);
	CHECK_NULLPTR(hdlTex, "texture handle", false);
	CHECK_NULLPTR(hdlMips, "texture max MIPMaps handle", false);

	// Give the texture a special name to avoid conflicts with regularly loaded textures
	std::string textureName = std::string(path) + std::string("##DISPLACEMENT_MAP##");
	std::string textureMipsName = std::string(path) + std::string("##DISPLACEMENT_MAP_MIPS##");

	// Check if the texture is already loaded
	auto hdl = s_world.find_texture(textureName);
	if(hdl != nullptr) {
		// Also find the max mipmaps
		auto hdlMaxMips = s_world.find_texture(textureMipsName);
		if(hdlMaxMips == nullptr) {
			logError("[", FUNCTION_NAME, "] Failed to find max MIPMaps for existing displacement map");
			return false;
		}
		s_world.ref_texture(hdl);
		s_world.ref_texture(hdlMaxMips);
		*hdlTex = hdl;
		*hdlMips = hdlMaxMips;
		return true;
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
		return false;
	}

	TextureHandle dispMap;
	if(texData.format != TextureFormat::FORMAT_R32F) {
		// Create a texture which will serve as a copy mechanism
		textures::CpuTexture tempTex(texData.width, texData.height, texData.layers,
									 static_cast<textures::Format>(texData.format),
									 textures::SamplingMode::NEAREST,
									 textures::MipmapType::NONE, texData.sRgb, false,
									 std::unique_ptr<u8[]>(texData.data));
		// Create an empty texture that we'll fill with the desired format
		std::unique_ptr<u8[]> texBuffer = std::make_unique<u8[]>(texData.width * texData.height * texData.layers * textures::PIXEL_SIZE(tempTex.get_format())
																 * (1 + ei::ilog2(std::max(texData.width, texData.height))));
		auto finalTex = std::make_unique<textures::Texture>(std::move(textureName), texData.width, texData.height, texData.layers,
															textures::MipmapType::MIN, textures::Format::R32F,
															textures::SamplingMode::NEAREST,
															texData.sRgb, true, std::move(texBuffer));
		auto cpuFinalTex = finalTex->template acquire<Device::CPU>();
		for(u32 layer = 0u; layer < texData.layers; ++layer) {
			for(u32 y = 0u; y < texData.height; ++y) {
				for(u32 x = 0u; x < texData.width; ++x) {
					const Pixel texel{ x, y };
					cpuFinalTex->write(tempTex.read(texel, layer), texel, layer);
				}
			}
		}
		cpuFinalTex->recompute_mipmaps(textures::MipmapType::MIN);
		finalTex->mark_changed(Device::CPU);
		dispMap = s_world.add_texture(std::move(finalTex));
	} else {
		auto finalTex = std::make_unique<textures::Texture>(std::move(textureName), texData.width, texData.height, texData.layers,
															textures::MipmapType::MIN, textures::Format::R32F,
															textures::SamplingMode::NEAREST,
															texData.sRgb, false, std::unique_ptr<u8[]>(texData.data));
		dispMap = s_world.add_texture(std::move(finalTex));
	}
	*hdlTex = dispMap;

	// Create the second texture holding only mipmaps (max. ones this time)
	// For that we need to compute one mipmap level ourselves
	if(texData.width > 1 || texData.height > 1) {
		std::unique_ptr<u8[]> texBuffer = std::make_unique<u8[]>(texData.width / 2 * texData.height / 2 * texData.layers
																 * textures::PIXEL_SIZE(textures::Format::R32F)
																 * ei::ilog2(std::max(texData.width, texData.height)));
		auto mipTex = std::make_unique<textures::Texture>(std::move(textureMipsName), texData.width / 2, texData.height / 2, texData.layers,
														  textures::MipmapType::MAX, textures::Format::R32F,
														  textures::SamplingMode::NEAREST,
														  texData.sRgb, true, std::move(texBuffer));
		auto cpuDispTech = dispMap->template acquire<Device::CPU>();
		auto cpuMipTech = mipTex->template acquire<Device::CPU>();
		for(u32 layer = 0u; layer < texData.layers; ++layer) {
			for(u32 y = 0u; y < texData.height; y += 2) {
				for(u32 x = 0u; x < texData.width; x += 2) {
					const Pixel texel{ x, y };
					ei::Vec4 val = cpuDispTech->read(Pixel{ x, y }, layer);
					val = std::max(val, cpuDispTech->read(Pixel{ x + 1, y }, layer));
					val = std::max(val, cpuDispTech->read(Pixel{ x , y + 1 }, layer));
					val = std::max(val, cpuDispTech->read(Pixel{ x + 1, y + 1 }, layer));
					cpuMipTech->write(val, Pixel{ x / 2, y / 2 }, layer);
				}
			}
		}
		cpuMipTech->recompute_mipmaps(textures::MipmapType::MAX);
		*hdlMips = s_world.add_texture(std::move(mipTex));
	}

	return true;
	CATCH_ALL(false)
}

const char* world_get_texture_name(TextureHdl hdl) {
	TRY
	CHECK_NULLPTR(hdl, "texture handle", nullptr);
	auto tex = static_cast<TextureHandle>(hdl);
	return tex->get_name().c_str();
	CATCH_ALL(nullptr)
}

Boolean world_get_texture_size(TextureHdl hdl, IVec2* size) {
	TRY
	CHECK_NULLPTR(hdl, "texture handle", false);
	auto tex = static_cast<TextureHandle>(hdl);
	(*size).x = tex->get_width();
	(*size).y = tex->get_height();
	return true;
	CATCH_ALL(false)
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
	StringView name = static_cast<const cameras::Camera*>(cam)->get_name();
	return &name[0];
	CATCH_ALL(nullptr)
}

Boolean world_get_camera_path_segment_count(ConstCameraHdl cam, uint32_t* segments) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	if(segments != nullptr) {
		*segments = static_cast<const cameras::Camera*>(cam)->get_path_segment_count();
	}
	return true;
	CATCH_ALL(false)
}

Boolean world_get_camera_position(ConstCameraHdl cam, Vec3* pos, const std::uint32_t pathIndex) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	if(pos != nullptr) {
		const auto& camera = *static_cast<const cameras::Camera*>(cam);
		*pos = util::pun<Vec3>(camera.get_position(std::min(pathIndex, camera.get_path_segment_count() - 1u)));
	}
	return true;
	CATCH_ALL(false)
}

Boolean world_get_camera_current_position(ConstCameraHdl cam, Vec3* pos) {
	TRY
		CHECK_NULLPTR(cam, "camera handle", false);
	if(pos != nullptr) {
		const auto& camera = *static_cast<const cameras::Camera*>(cam);
		*pos = util::pun<Vec3>(camera.get_position(std::min(s_world.get_frame_current() - s_world.get_frame_start(),
															camera.get_path_segment_count() - 1u)));
	}
	return true;
	CATCH_ALL(false)
}

Boolean world_get_camera_direction(ConstCameraHdl cam, Vec3* dir, const std::uint32_t pathIndex) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	if(dir != nullptr) {
		const auto& camera = *static_cast<const cameras::Camera*>(cam);
		*dir = util::pun<Vec3>(camera.get_view_dir(std::min(pathIndex, camera.get_path_segment_count() - 1u)));
	}
	return true;
	CATCH_ALL(false)
}

Boolean world_get_camera_current_direction(ConstCameraHdl cam, Vec3* dir) {
	TRY
		CHECK_NULLPTR(cam, "camera handle", false);
	if(dir != nullptr) {
		const auto& camera = *static_cast<const cameras::Camera*>(cam);
		*dir = util::pun<Vec3>(camera.get_view_dir(std::min(s_world.get_frame_current() - s_world.get_frame_start(),
															camera.get_path_segment_count() - 1u)));
	}
	return true;
	CATCH_ALL(false)
}

Boolean world_get_camera_up(ConstCameraHdl cam, Vec3* up, const std::uint32_t pathIndex) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	if(up != nullptr) {
		const auto& camera = *static_cast<const cameras::Camera*>(cam);
		*up = util::pun<Vec3>(camera.get_up_dir(std::min(pathIndex, camera.get_path_segment_count() - 1u)));
	}
	return true;
	CATCH_ALL(false)
}

Boolean world_get_camera_current_up(ConstCameraHdl cam, Vec3* up) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	if(up != nullptr) {
		const auto& camera = *static_cast<const cameras::Camera*>(cam);
		*up = util::pun<Vec3>(camera.get_up_dir(std::min(s_world.get_frame_current() - s_world.get_frame_start(),
															camera.get_path_segment_count() - 1u)));
	}
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

Boolean world_set_camera_position(CameraHdl cam, Vec3 pos, const std::uint32_t pathIndex) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	auto& camera = *static_cast<cameras::Camera*>(cam);
	camera.set_position(util::pun<scene::Point>(pos), std::min(pathIndex, camera.get_path_segment_count() - 1u));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_camera_current_position(CameraHdl cam, Vec3 pos) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	auto& camera = *static_cast<cameras::Camera*>(cam);
	camera.set_position(util::pun<scene::Point>(pos), std::min(s_world.get_frame_current() - s_world.get_frame_start(),
															   camera.get_path_segment_count() - 1u));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_camera_direction(CameraHdl cam, Vec3 dir, Vec3 up, const std::uint32_t pathIndex) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	auto& camera = *static_cast<cameras::Camera*>(cam);
	camera.set_view_dir(util::pun<scene::Direction>(dir), util::pun<scene::Direction>(up), pathIndex);
	camera.set_view_dir(util::pun<scene::Direction>(dir), util::pun<scene::Direction>(up),
						std::min(pathIndex, camera.get_path_segment_count() - 1u));
	// TODO: compute proper rotation
	return true;
	CATCH_ALL(false)
}

Boolean world_set_camera_current_direction(CameraHdl cam, Vec3 dir, Vec3 up) {
	TRY
		CHECK_NULLPTR(cam, "camera handle", false);
	auto& camera = *static_cast<cameras::Camera*>(cam);
	camera.set_view_dir(util::pun<scene::Direction>(dir), util::pun<scene::Direction>(up),
						std::min(s_world.get_frame_current() - s_world.get_frame_start(),
								 camera.get_path_segment_count() - 1u));
	// TODO: compute proper rotation
	return true;
	CATCH_ALL(false)
}

Boolean world_set_camera_near(CameraHdl cam, float near) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	CHECK(near > 0.f, "near-plane", false);
	static_cast<cameras::Camera*>(cam)->set_near(near);
	return true;
	CATCH_ALL(false)
}

Boolean world_set_camera_far(CameraHdl cam, float far) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	CHECK(far > 0.f, "far-plane", false);
	static_cast<cameras::Camera*>(cam)->set_far(far);
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
		*aperture =  camera.get_aperture_in_f_stops();
	return true;
	CATCH_ALL(false)
}

Boolean world_set_focus_camera_focal_length(CameraHdl cam, float focalLength) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	CHECK(focalLength > 0.f, "focalLength", false);
	static_cast<cameras::Focus*>(cam)->set_focal_length(focalLength);
	return true;
	CATCH_ALL(false)
}

Boolean world_set_focus_camera_focus_distance(CameraHdl cam, float focusDistance) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	CHECK(focusDistance > 0.f, "focus distance", false);
	static_cast<cameras::Focus*>(cam)->set_focus_distance(focusDistance);
	return true;
	CATCH_ALL(false)
}

Boolean world_set_focus_camera_sensor_height(CameraHdl cam, float sensorHeight) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	CHECK(sensorHeight > 0.f, "sensor height", false);
	static_cast<cameras::Focus*>(cam)->set_sensor_height(sensorHeight);
	return true;
	CATCH_ALL(false)
}

Boolean world_set_focus_camera_aperture(CameraHdl cam, float aperture) {
	TRY
	CHECK_NULLPTR(cam, "camera handle", false);
	CHECK(aperture > 0.f, "aperture", false);
	auto& camera = *static_cast<cameras::Focus*>(cam);
	camera.set_aperture_in_f_stops(aperture);
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

LodLevel scenario_get_global_lod_level(ScenarioHdl scenario) {
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

Boolean world_set_frame_current(const uint32_t animationFrame) {
	TRY
	// Necessary to lock because setting a new frame clears out the scene!
	auto lock = std::scoped_lock(s_iterationMutex);
	const u32 oldFrame = s_world.get_frame_current();
	if(s_currentRenderer != nullptr)
		s_currentRenderer->on_animation_frame_changing(oldFrame, animationFrame);
	if(s_world.set_frame_current(animationFrame))
		for(std::size_t i = 0u; i < s_renderers.size(); ++i)
			for(auto& renderer : s_renderers.get(i))
				renderer->on_animation_frame_changed(oldFrame, animationFrame);
	return true;
	CATCH_ALL(false)
}

Boolean world_get_frame_current(uint32_t* animationFrame) {
	TRY
	if(animationFrame != nullptr)
		*animationFrame = s_world.get_frame_current();
	return true;
	CATCH_ALL(false)
}

Boolean world_get_frame_start(uint32_t* frameStart) {
	TRY
	if(frameStart != nullptr)
		*frameStart = s_world.get_frame_start();
	return true;
	CATCH_ALL(false)
}

Boolean world_get_frame_end(uint32_t* frameEnd) {
	TRY
	if(frameEnd != nullptr)
		*frameEnd = s_world.get_frame_end();
	return true;
	CATCH_ALL(false)
}

void world_set_tessellation_level(const float maxTessLevel) {
	s_world.set_tessellation_level(maxTessLevel);
}

float world_get_tessellation_level() {
	return s_world.get_tessellation_level();
}

Boolean scenario_set_camera(ScenarioHdl scenario, CameraHdl cam) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	static_cast<Scenario*>(scenario)->set_camera(static_cast<CameraHandle>(cam));
	if(scenario == world_get_current_scenario() && s_world.get_current_scene() != nullptr)
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

Boolean scenario_mask_instance(ScenarioHdl scenario, InstanceHdl inst) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(inst, "instance handle", false);
	static_cast<Scenario*>(scenario)->mask_instance(static_cast<const Instance*>(inst));
	return true;
	CATCH_ALL(false)
}

Boolean scenario_set_object_tessellation_level(ScenarioHdl scenario, ObjectHdl hdl, float level) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(hdl, "object handle", false);
	Scenario& scen = *static_cast<Scenario*>(scenario);
	scen.set_tessellation_level(static_cast<ConstObjectHandle>(hdl), level);
	return true;
	CATCH_ALL(false)
}

Boolean scenario_set_object_adaptive_tessellation(ScenarioHdl scenario, ObjectHdl hdl, Boolean value) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(hdl, "object handle", false);
	Scenario& scen = *static_cast<Scenario*>(scenario);
	scen.set_adaptive_tessellation(static_cast<ConstObjectHandle>(hdl), value);
	return true;
	CATCH_ALL(false)
}

Boolean scenario_set_object_phong_tessellation(ScenarioHdl scenario, ObjectHdl hdl, Boolean value) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(hdl, "object handle", false);
	Scenario& scen = *static_cast<Scenario*>(scenario);
	scen.set_phong_tessellation(static_cast<ConstObjectHandle>(hdl), value);
	return true;
	CATCH_ALL(false)
}

Boolean scenario_has_object_tessellation_info(ScenarioHdl scenario, ObjectHdl hdl, Boolean* value) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(hdl, "object handle", false);
	const Scenario& scen = *static_cast<const Scenario*>(scenario);
	if(value)
		*value = scen.get_tessellation_info(static_cast<ConstObjectHandle>(hdl)).has_value();
	return true;
	CATCH_ALL(false)
}

Boolean scenario_get_object_tessellation_level(ScenarioHdl scenario, ObjectHdl hdl, float* level) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(hdl, "object handle", false);
	const Scenario& scen = *static_cast<const Scenario*>(scenario);
	const auto info = scen.get_tessellation_info(static_cast<ConstObjectHandle>(hdl));
	CHECK(info, "object tessellation info", false);
	if(level)
		*level = info->level.value_or(s_world.get_tessellation_level());
	return true;
	CATCH_ALL(false)
}

Boolean scenario_get_object_adaptive_tessellation(ScenarioHdl scenario, ObjectHdl hdl, Boolean* value) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(hdl, "object handle", false);
	const Scenario& scen = *static_cast<const Scenario*>(scenario);
	const auto info = scen.get_tessellation_info(static_cast<ConstObjectHandle>(hdl));
	CHECK(info, "object tessellation info", false);
	if(value)
		*value = info->adaptive;
	return true;
	CATCH_ALL(false)
}

Boolean scenario_get_object_phong_tessellation(ScenarioHdl scenario, ObjectHdl hdl, Boolean* value) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(hdl, "object handle", false);
	const Scenario& scen = *static_cast<const Scenario*>(scenario);
	const auto info = scen.get_tessellation_info(static_cast<ConstObjectHandle>(hdl));
	CHECK(info, "object tessellation info", false);
	if(value)
		*value = info->usePhong;
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

Boolean scenario_set_instance_lod(ScenarioHdl scenario, InstanceHdl inst,
							 LodLevel level) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(inst, "instance handle", false);
	static_cast<Scenario*>(scenario)->set_custom_lod(static_cast<const Instance*>(inst),
													 level);
	return true;
	CATCH_ALL(false)
}

IndexType scenario_get_point_light_count(ScenarioHdl scenario) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_INDEX);
	return static_cast<IndexType>(static_cast<const Scenario*>(scenario)->get_point_lights().size());
	CATCH_ALL(INVALID_INDEX)
}

IndexType scenario_get_spot_light_count(ScenarioHdl scenario) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_INDEX);
	return static_cast<IndexType>(static_cast<const Scenario*>(scenario)->get_spot_lights().size());
	CATCH_ALL(INVALID_INDEX)
}

IndexType scenario_get_dir_light_count(ScenarioHdl scenario) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_INDEX);
	return static_cast<IndexType>(static_cast<const Scenario*>(scenario)->get_dir_lights().size());
	CATCH_ALL(INVALID_INDEX)
}

Boolean scenario_has_envmap_light(ScenarioHdl scenario) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	const Scenario& scen = *static_cast<const Scenario*>(scenario);
	return (s_world.get_background(scen.get_background()) != &s_world.get_default_background());
	CATCH_ALL(INVALID_INDEX)
}

LightHdl scenario_get_light_handle(ScenarioHdl scenario, IndexType index, LightType type) {
	const LightHdl invalid{ LightType::LIGHT_COUNT, 0x1FFFFFFF };
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", invalid);
	const Scenario& scen = *static_cast<const Scenario*>(scenario);
	switch(type) {
		case LightType::LIGHT_POINT:
			if(index >= static_cast<IndexType>(scen.get_point_lights().size())) {
				logError("[", FUNCTION_NAME, "] Point light index out of bounds (",
						 index, " >= ", scen.get_point_lights().size(), ")");
				return invalid;
			}
			return LightHdl{ LightType::LIGHT_POINT, scen.get_point_lights()[index] };
			break;
		case LightType::LIGHT_SPOT:
			if(index >= static_cast<IndexType>(scen.get_spot_lights().size())) {
				logError("[", FUNCTION_NAME, "] Spot light index out of bounds (",
						 index, " >= ", scen.get_spot_lights().size(), ")");
				return invalid;
			}
			return LightHdl{ LightType::LIGHT_SPOT, scen.get_spot_lights()[index] };
			break;
		case LightType::LIGHT_DIRECTIONAL:
			if(index >= static_cast<IndexType>(scen.get_dir_lights().size())) {
				logError("[", FUNCTION_NAME, "] Directional light index out of bounds (",
						 index, " >= ", scen.get_dir_lights().size(), ")");
				return invalid;
			}
			return LightHdl{ LightType::LIGHT_DIRECTIONAL, scen.get_dir_lights()[index] };
			break;
		case LightType::LIGHT_ENVMAP:
			if(index >= 1) {
				logError("[", FUNCTION_NAME, "] Background index out of bounds (", index, " >= 1)");
				return invalid;
			}
			return LightHdl{ LightType::LIGHT_ENVMAP, scen.get_background() };
			break;
		default:
			logError("[", FUNCTION_NAME, "] Unknown light type");
			return invalid;
	}
	CATCH_ALL(invalid)
}

Boolean scenario_add_light(ScenarioHdl scenario, LightHdl hdl) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	Scenario& scen = *static_cast<Scenario*>(scenario);

	switch(hdl.type) {
		case LightType::LIGHT_POINT: {
			if(s_world.get_point_light(hdl.index, 0u) == nullptr) {
				logError("[", FUNCTION_NAME, "] Invalid point light handle");
				return false;
			}
			scen.add_point_light(hdl.index);
			if(s_currentRenderer != nullptr && s_currentRenderer->has_scene() && scenario == s_world.get_current_scenario())
				s_currentRenderer->on_light_changed();
		}	break;
		case LightType::LIGHT_SPOT: {
			if(s_world.get_spot_light(hdl.index, 0u) == nullptr) {
				logError("[", FUNCTION_NAME, "] Invalid spot light handle");
				return false;
			}
			scen.add_spot_light(hdl.index);
			if(s_currentRenderer != nullptr && s_currentRenderer->has_scene() && scenario == s_world.get_current_scenario())
				s_currentRenderer->on_light_changed();
		}	break;
		case LightType::LIGHT_DIRECTIONAL: {
			if(s_world.get_dir_light(hdl.index, 0u) == nullptr) {
				logError("[", FUNCTION_NAME, "] Invalid directional light handle");
				return false;
			}
			scen.add_dir_light(hdl.index);
			if(s_currentRenderer != nullptr && s_currentRenderer->has_scene() && scenario == s_world.get_current_scenario())
				s_currentRenderer->on_light_changed();
		}	break;
		case LightType::LIGHT_ENVMAP: {
			lights::Background* light = s_world.get_background(hdl.index);
			if(s_world.get_background(hdl.index) == nullptr) {
				logError("[", FUNCTION_NAME, "] Invalid background light handle");
				return false;
			}
			if(light->get_type() == lights::BackgroundType::ENVMAP && s_world.get_background(scen.get_background())->get_type() == lights::BackgroundType::ENVMAP) {
				logWarning("[", FUNCTION_NAME, "] The scenario already has an environment light; overwriting '",
						   s_world.get_light_name(scen.get_background(), lights::LightType::ENVMAP_LIGHT), "' with '",
						   s_world.get_light_name(hdl.index, lights::LightType::ENVMAP_LIGHT), "'");
			}
			scen.set_background(hdl.index);
			if(s_currentRenderer != nullptr && s_currentRenderer->has_scene() && scenario == s_world.get_current_scenario())
				s_currentRenderer->on_light_changed();
		}	break;
		default:
			logError("[", FUNCTION_NAME, "] Unknown or invalid light type");
			return false;
	}
	return true;
	CATCH_ALL(false)
}

Boolean scenario_remove_light(ScenarioHdl scenario, LightHdl hdl) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	Scenario& scen = *static_cast<Scenario*>(scenario);
	// TODO: don't remove the light if it's the last remaining one

	switch(hdl.type) {
		case LightType::LIGHT_POINT: {
			scen.remove_point_light(hdl.index);
			if(s_currentRenderer != nullptr && s_currentRenderer->has_scene() && scenario == s_world.get_current_scenario())
				s_currentRenderer->on_light_changed();
		}	break;
		case LightType::LIGHT_SPOT: {
			scen.remove_spot_light(hdl.index);
			if(s_currentRenderer != nullptr && s_currentRenderer->has_scene() && scenario == s_world.get_current_scenario())
				s_currentRenderer->on_light_changed();
		}	break;
		case LightType::LIGHT_DIRECTIONAL: {
			scen.remove_dir_light(hdl.index);
			if(s_currentRenderer != nullptr && s_currentRenderer->has_scene() && scenario == s_world.get_current_scenario())
				s_currentRenderer->on_light_changed();
		}	break;
		case LightType::LIGHT_ENVMAP: {
			// Make sure we only remove the background from the scenario if it is actually active
			if(scen.get_background() == hdl.index) {
				scen.remove_background();
				if(s_currentRenderer != nullptr && s_currentRenderer->has_scene() && scenario == s_world.get_current_scenario())
					s_currentRenderer->on_light_changed();
			}
		}	break;
		default:
			logError("[", FUNCTION_NAME, "] Unknown or invalid light type");
			return false;
	}
	return true;
	CATCH_ALL(false)
}

void scenario_reserve_material_slots(ScenarioHdl scenario, size_t count) {
	TRY
	static_cast<Scenario*>(scenario)->reserve_material_slots(count);
	CATCH_ALL(;)
}

void scenario_reserve_custom_object_properties(ScenarioHdl scenario, size_t objects) {
	TRY
		static_cast<Scenario*>(scenario)->reserve_custom_object_properties(objects);
	CATCH_ALL(;)
}

void scenario_reserve_custom_instance_properties(ScenarioHdl scenario, size_t instances) {
	TRY
		static_cast<Scenario*>(scenario)->reserve_custom_instance_properties(instances);
	CATCH_ALL(;)
}

MatIdx scenario_declare_material_slot(ScenarioHdl scenario,
									  const char* name, std::size_t nameLength) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_MATERIAL);
	CHECK_NULLPTR(name, "material name", INVALID_MATERIAL);
	StringView nameView(name, std::min<std::size_t>(nameLength, std::strlen(name)));
	return static_cast<Scenario*>(scenario)->declare_material_slot(nameView);
	CATCH_ALL(INVALID_MATERIAL)
}

MatIdx scenario_get_material_slot(ScenarioHdl scenario,
								  const char* name, std::size_t nameLength) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_MATERIAL);
	CHECK_NULLPTR(name, "material name", INVALID_MATERIAL);
	StringView nameView(name, std::min<std::size_t>(nameLength, std::strlen(name)));
	return static_cast<const Scenario*>(scenario)->get_material_slot_index(nameView);
	CATCH_ALL(INVALID_MATERIAL)
}

const char* scenario_get_material_slot_name(ScenarioHdl scenario, MatIdx slot) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	return static_cast<const Scenario*>(scenario)->get_material_slot_name(slot).data();
	CATCH_ALL(nullptr)
}
size_t scenario_get_material_slot_count(ScenarioHdl scenario){
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", 0);
	return static_cast<const Scenario*>(scenario)->get_num_material_slots();
	CATCH_ALL(0)
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

Boolean world_finalize_scenario(ConstScenarioHdl scenario, const char** msg) {
	TRY
	CHECK_NULLPTR(scenario, "scenario handle", false);
	switch(s_world.finalize_scenario(static_cast<ConstScenarioHandle>(scenario))) {
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
	auto& camera = *s_world.get_current_scenario()->get_camera();
	camera.move(x, y, z, std::min(camera.get_path_segment_count() - 1u, s_world.get_frame_current() - s_world.get_frame_start()));
	return true;
	CATCH_ALL(false)
}

Boolean scene_rotate_active_camera(float x, float y, float z) {
	TRY
	if(s_world.get_current_scene() == nullptr) {
		logError("[", FUNCTION_NAME, "] No scene loaded yet");
		return false;
	}
	auto& camera = *s_world.get_current_scenario()->get_camera();
	const u32 frame = std::min(camera.get_path_segment_count() - 1u, s_world.get_frame_current() - s_world.get_frame_start());
	camera.rotate_up_down(x, frame);
	camera.rotate_left_right(y, frame);
	camera.roll(z, frame);
	return true;
	CATCH_ALL(false)
}

Boolean scene_is_sane() {
	TRY
	ConstSceneHandle sceneHdl = nullptr;
	if(!s_world.is_current_scene_valid()) {
		// Check if a rebuild was demanded
		if(s_world.get_current_scenario() != nullptr) {
			world_load_scenario(s_world.get_current_scenario());
			sceneHdl = s_world.get_current_scene();
		}
	} else {
		sceneHdl = s_world.get_current_scene();
	}

	if(sceneHdl != nullptr)
		return sceneHdl->is_sane();
	return false;
	CATCH_ALL(false)
}

Boolean scene_request_retessellation() {
	TRY
	auto lock = std::scoped_lock(s_iterationMutex);
	if(s_currentRenderer != nullptr)
		s_currentRenderer->on_tessellation_changing();
	s_world.retessellate();
	if(s_currentRenderer != nullptr)
		s_currentRenderer->on_tessellation_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_get_point_light_position(ConstLightHdl hdl, Vec3* pos, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_POINT, "light type must be point", false);
	const lights::PointLight* light = s_world.get_point_light(hdl.index, frame);
	CHECK_NULLPTR(light, "point light handle", false);
	if(pos != nullptr)
		*pos = util::pun<Vec3>(light->position);
	return true;
	CATCH_ALL(false)
}

Boolean world_get_point_light_intensity(ConstLightHdl hdl, Vec3* intensity, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_POINT, "light type must be point", false);
	const lights::PointLight* light = s_world.get_point_light(hdl.index, frame);
	CHECK_NULLPTR(light, "point light handle", false);
	if(intensity != nullptr)
		*intensity = util::pun<Vec3>(light->intensity);
	return true;
	CATCH_ALL(false)
}

Boolean world_get_point_light_path_segments(ConstLightHdl hdl, uint32_t* segments) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_POINT, "light type must be point", false);
	if(segments != nullptr)
		*segments = static_cast<uint32_t>(s_world.get_point_light_segment_count(hdl.index));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_point_light_position(LightHdl hdl, Vec3 pos, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_POINT, "light type must be point", false);
	auto lock = std::scoped_lock(s_iterationMutex);
	lights::PointLight* light = s_world.get_point_light(hdl.index, frame);
	CHECK_NULLPTR(light, "point light handle", false);
	light->position = util::pun<ei::Vec3>(pos);
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::POINT_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_set_point_light_intensity(LightHdl hdl, Vec3 intensity, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_POINT, "light type must be point", false);
	auto lock = std::scoped_lock(s_iterationMutex);
	lights::PointLight* light = s_world.get_point_light(hdl.index, frame);
	CHECK_NULLPTR(light, "point light handle", false);
	light->intensity = util::pun<ei::Vec3>(intensity);
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::POINT_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_get_spot_light_path_segments(ConstLightHdl hdl, uint32_t* segments) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	if(segments != nullptr)
		*segments = static_cast<uint32_t>(s_world.get_spot_light_segment_count(hdl.index));
	return true;
	CATCH_ALL(false)
}

Boolean world_get_spot_light_position(ConstLightHdl hdl, Vec3* pos, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	const lights::SpotLight* light = s_world.get_spot_light(hdl.index, frame);
	CHECK_NULLPTR(light, "spot light handle", false);
	if(pos != nullptr)
		*pos = util::pun<Vec3>(light->position);
	return true;
	CATCH_ALL(false)
}

Boolean world_get_spot_light_intensity(ConstLightHdl hdl, Vec3* intensity, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	const lights::SpotLight* light = s_world.get_spot_light(hdl.index, frame);
	CHECK_NULLPTR(light, "spot light handle", false);
	if(intensity != nullptr)
		*intensity = util::pun<Vec3>(light->intensity);
	return true;
	CATCH_ALL(false)
}

Boolean world_get_spot_light_direction(ConstLightHdl hdl, Vec3* direction, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	const lights::SpotLight* light = s_world.get_spot_light(hdl.index, frame);
	CHECK_NULLPTR(light, "spot light handle", false);
	if(direction != nullptr)
		*direction = util::pun<Vec3>(light->direction);
	return true;
	CATCH_ALL(false)
}

Boolean world_get_spot_light_angle(ConstLightHdl hdl, float* angle, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	const lights::SpotLight* light = s_world.get_spot_light(hdl.index, frame);
	CHECK_NULLPTR(light, "spot light handle", false);
	if(angle != nullptr)
		*angle = std::acos(__half2float(light->cosThetaMax));
	return true;
	CATCH_ALL(false)
}

Boolean world_get_spot_light_falloff(ConstLightHdl hdl, float* falloff, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	const lights::SpotLight* light = s_world.get_spot_light(hdl.index, frame);
	CHECK_NULLPTR(light, "spot light handle", false);
	if(falloff != nullptr)
		*falloff = std::acos(__half2float(light->cosFalloffStart));
	return true;
	CATCH_ALL(false)
}

Boolean world_set_spot_light_position(LightHdl hdl, Vec3 pos, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	auto lock = std::scoped_lock(s_iterationMutex);
	lights::SpotLight* light = s_world.get_spot_light(hdl.index, frame);
	CHECK_NULLPTR(light, "spot light handle", false);
	light->position = util::pun<ei::Vec3>(pos);
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::SPOT_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_set_spot_light_intensity(LightHdl hdl, Vec3 intensity, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	auto lock = std::scoped_lock(s_iterationMutex);
	lights::SpotLight* light = s_world.get_spot_light(hdl.index, frame);
	CHECK_NULLPTR(light, "spot light handle", false);
	light->intensity = util::pun<ei::Vec3>(intensity);
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::SPOT_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_set_spot_light_direction(LightHdl hdl, Vec3 direction, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	auto lock = std::scoped_lock(s_iterationMutex);
	lights::SpotLight* light = s_world.get_spot_light(hdl.index, frame);
	CHECK_NULLPTR(light, "spot light handle", false);
	ei::Vec3 actualDirection = ei::normalize(util::pun<ei::Vec3>(direction));
	if(!ei::approx(ei::len(actualDirection), 1.0f)) {
		logError("[", FUNCTION_NAME, "] Spotlight direction cannot be a null vector");
		return false;
	}
	light->direction = util::pun<ei::Vec3>(actualDirection);
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::SPOT_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_set_spot_light_angle(LightHdl hdl, float angle, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	lights::SpotLight* light = s_world.get_spot_light(hdl.index, frame);
	CHECK_NULLPTR(light, "spot light handle", false);
	auto lock = std::scoped_lock(s_iterationMutex);

	float actualAngle = std::fmod(angle, 2.f * ei::PI);
	if(actualAngle < 0.f)
		actualAngle += 2.f*ei::PI;
	if(actualAngle > ei::PI / 2.f) {
		logWarning("[", FUNCTION_NAME, "] Spotlight angle will be clamped between 0-180 degrees");
		actualAngle = ei::PI / 2.f;
	}
	light->cosThetaMax = __float2half(std::cos(actualAngle));
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::SPOT_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_set_spot_light_falloff(LightHdl hdl, float falloff, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_SPOT, "light type must be spot", false);
	lights::SpotLight* light = s_world.get_spot_light(hdl.index, frame);
	CHECK_NULLPTR(light, "spot light handle", false);
	auto lock = std::scoped_lock(s_iterationMutex);
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
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::SPOT_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_get_dir_light_path_segments(ConstLightHdl hdl, uint32_t* segments) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_DIRECTIONAL, "light type must be directional", false);
	if(segments != nullptr)
		*segments = static_cast<uint32_t>(s_world.get_dir_light_segment_count(hdl.index));
	return true;
	CATCH_ALL(false)
}

Boolean world_get_dir_light_direction(ConstLightHdl hdl, Vec3* direction, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_DIRECTIONAL, "light type must be directional", false);
	const lights::DirectionalLight* light = s_world.get_dir_light(hdl.index, frame);
	CHECK_NULLPTR(light, "directional light handle", false);
	if(direction != nullptr)
		*direction = util::pun<Vec3>(light->direction);
	return true;
	CATCH_ALL(false)
}

Boolean world_get_dir_light_irradiance(ConstLightHdl hdl, Vec3* irradiance, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_DIRECTIONAL, "light type must be directional", false);
	const lights::DirectionalLight* light = s_world.get_dir_light(hdl.index, frame);
	CHECK_NULLPTR(light, "directional light handle", false);
	if(irradiance != nullptr)
		*irradiance = util::pun<Vec3>(light->irradiance);
	return true;
	CATCH_ALL(false)
}

Boolean world_set_dir_light_direction(LightHdl hdl, Vec3 direction, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_DIRECTIONAL, "light type must be directional", false);
	lights::DirectionalLight* light = s_world.get_dir_light(hdl.index, frame);
	CHECK_NULLPTR(light, "directional light handle", false);
	auto lock = std::scoped_lock(s_iterationMutex);
	ei::Vec3 actualDirection = ei::normalize(util::pun<ei::Vec3>(direction));
	if(!ei::approx(ei::len(actualDirection), 1.0f)) {
		logError("[", FUNCTION_NAME, "] Directional light direction cannot be a null vector");
		return false;
	}
	light->direction = util::pun<ei::Vec3>(actualDirection);
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::DIRECTIONAL_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_set_dir_light_irradiance(LightHdl hdl, Vec3 irradiance, const uint32_t frame) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_DIRECTIONAL, "light type must be directional", false);
	lights::DirectionalLight* light = s_world.get_dir_light(hdl.index, frame);
	CHECK_NULLPTR(light, "directional light handle", false);
	auto lock = std::scoped_lock(s_iterationMutex);
	light->irradiance = util::pun<ei::Vec3>(irradiance);
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::DIRECTIONAL_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

const char* world_get_env_light_map(ConstLightHdl hdl) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be envmap", nullptr);
	const lights::Background* background = s_world.get_background(hdl.index);
	if(background == nullptr || background->get_type() != lights::BackgroundType::ENVMAP) {
		logError("[", FUNCTION_NAME, "] The background is not an environment-mapped light");
		return nullptr;
	}
	ConstTextureHandle envmap = background->get_envmap();
	CHECK_NULLPTR(envmap, "environment-mapped light handle", nullptr);
	return envmap->get_name().c_str();
	CATCH_ALL(nullptr)
}

Boolean world_get_env_light_scale(LightHdl hdl, Vec3* color) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be envmap", false);
	const lights::Background* background = s_world.get_background(hdl.index);
	if(background == nullptr) {
		logError("[", FUNCTION_NAME, "] The background is not an environment-mapped light");
		return false;
	}
	CHECK_NULLPTR(background, "environment-mapped light handle", false);
	if(color)
		*color = util::pun<Vec3>(background->get_scale());
	return true;
	CATCH_ALL(false)
}

Boolean world_get_sky_light_turbidity(LightHdl hdl, float* turbidity) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be sky", false);
	const lights::Background* background = s_world.get_background(hdl.index);
	if(background == nullptr || background->get_type() != lights::BackgroundType::SKY_HOSEK) {
		logError("[", FUNCTION_NAME, "] The background is not a sky light");
		return false;
	}
	CHECK_NULLPTR(background, "sky light handle", false);
	if(turbidity)
		*turbidity = background->get_sky_turbidity();
	return true;
	CATCH_ALL(false)
}

Boolean world_set_sky_light_turbidity(LightHdl hdl, float turbidity) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be sky", false);
	lights::Background* background = s_world.get_background(hdl.index);
	if(background == nullptr || background->get_type() != lights::BackgroundType::SKY_HOSEK) {
		logError("[", FUNCTION_NAME, "] The background is not a sky light");
		return false;
	}
	CHECK_NULLPTR(background, "sky light handle", false);
	background->set_sky_turbidity(turbidity);
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::ENVMAP_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_get_sky_light_albedo(LightHdl hdl, float* albedo) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be sky", false);
	const lights::Background* background = s_world.get_background(hdl.index);
	if(background == nullptr || background->get_type() != lights::BackgroundType::SKY_HOSEK) {
		logError("[", FUNCTION_NAME, "] The background is not a sky light");
		return false;
	}
	CHECK_NULLPTR(background, "sky light handle", false);
	if(albedo)
		*albedo = background->get_sky_albedo();
	return true;
	CATCH_ALL(false)
}

Boolean world_set_sky_light_albedo(LightHdl hdl, float albedo) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be sky", false);
	lights::Background* background = s_world.get_background(hdl.index);
	if(background == nullptr || background->get_type() != lights::BackgroundType::SKY_HOSEK) {
		logError("[", FUNCTION_NAME, "] The background is not a sky light");
		return false;
	}
	CHECK_NULLPTR(background, "sky light handle", false);
	background->set_sky_albedo(albedo);
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::ENVMAP_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_get_sky_light_solar_radius(LightHdl hdl, float* radius) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be sky", false);
	const lights::Background* background = s_world.get_background(hdl.index);
	if(background == nullptr || background->get_type() != lights::BackgroundType::SKY_HOSEK) {
		logError("[", FUNCTION_NAME, "] The background is not a sky light");
		return false;
	}
	CHECK_NULLPTR(background, "sky light handle", false);
	if(radius)
		*radius = background->get_sky_solar_radius();
	return true;
	CATCH_ALL(false)
}

Boolean world_set_sky_light_solar_radius(LightHdl hdl, float radius) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be sky", false);
	lights::Background* background = s_world.get_background(hdl.index);
	if(background == nullptr || background->get_type() != lights::BackgroundType::SKY_HOSEK) {
		logError("[", FUNCTION_NAME, "] The background is not a sky light");
		return false;
	}
	CHECK_NULLPTR(background, "sky light handle", false);
	background->set_sky_solar_radius(radius);
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::ENVMAP_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_get_sky_light_sun_direction(LightHdl hdl, Vec3* sunDir) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be sky", false);
	const lights::Background* background = s_world.get_background(hdl.index);
	if(background == nullptr || background->get_type() != lights::BackgroundType::SKY_HOSEK) {
		logError("[", FUNCTION_NAME, "] The background is not a sky light");
		return false;
	}
	CHECK_NULLPTR(background, "sky light handle", false);
	if(sunDir)
		*sunDir = util::pun<Vec3>(background->get_sky_sun_direction());
	return true;
	CATCH_ALL(false)
}

Boolean world_set_sky_light_sun_direction(LightHdl hdl, Vec3 sunDir) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be sky", false);
	lights::Background* background = s_world.get_background(hdl.index);
	if(background == nullptr || background->get_type() != lights::BackgroundType::SKY_HOSEK) {
		logError("[", FUNCTION_NAME, "] The background is not a sky light");
		return false;
	}
	CHECK_NULLPTR(background, "sky light handle", false);
	background->set_sky_sun_direction(util::pun<ei::Vec3>(sunDir));
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::ENVMAP_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_get_env_light_color(LightHdl hdl, Vec3* color) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be monochrom", false);
	const lights::Background* background = s_world.get_background(hdl.index);
	if(background == nullptr || background->get_type() != lights::BackgroundType::COLORED) {
		logError("[", FUNCTION_NAME, "] The background is not a monochrom light");
		return false;
	}
	CHECK_NULLPTR(background, "background light handle", false);
	if(color)
		*color = util::pun<Vec3>(background->get_monochrom_color());
	return true;
	CATCH_ALL(false)
}

Boolean world_set_env_light_color(LightHdl hdl, Vec3 color) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be sky", false);
	lights::Background* background = s_world.get_background(hdl.index);
	if(background == nullptr || background->get_type() != lights::BackgroundType::COLORED) {
		logError("[", FUNCTION_NAME, "] The background is not a monochrom light");
		return false;
	}
	CHECK_NULLPTR(background, "background light handle", false);
	background->set_monochrom_color(util::pun<ei::Vec3>(color));
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::ENVMAP_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_set_env_light_map(LightHdl hdl, TextureHdl tex) {
	TRY
	CHECK_NULLPTR(tex, "texture handle", false);
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be envmap", false);
	auto lock = std::scoped_lock(s_iterationMutex);
	s_world.replace_envlight_texture(hdl.index, reinterpret_cast<TextureHandle>(tex));
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::ENVMAP_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

Boolean world_set_env_light_scale(LightHdl hdl, Vec3 color) {
	TRY
	CHECK(hdl.type == LightType::LIGHT_ENVMAP, "light type must be envmap", false);
	auto lock = std::scoped_lock(s_iterationMutex);
	lights::Background* background = s_world.get_background(hdl.index);
	if(background == nullptr) {
		logError("[", FUNCTION_NAME, "] The background is not an environment-mapped light");
		return false;
	}
	background->set_scale(util::pun<Spectrum>(color));
	if(s_world.mark_light_dirty(hdl.index, lights::LightType::ENVMAP_LIGHT) && s_currentRenderer != nullptr)
		s_currentRenderer->on_light_changed();
	return true;
	CATCH_ALL(false)
}

uint32_t render_get_renderer_count() {
	return static_cast<uint32_t>(s_renderers.size());
}

uint32_t render_get_renderer_variations(uint32_t index) {
	TRY
	CHECK(index < s_renderers.size(), "renderer index out of bounds", 0);
	return static_cast<uint32_t>(s_renderers.get(index).size());
	CATCH_ALL(0)
}

const char* render_get_renderer_name(uint32_t index) {
	TRY
	CHECK(index < s_renderers.size(), "renderer index out of bounds", nullptr);
	return &s_renderers.get_key(index)[0u];
	CATCH_ALL(nullptr)
}

const char* render_get_renderer_short_name(uint32_t index) {
	TRY
	CHECK(index < s_renderers.size(), "renderer index out of bounds", nullptr);
	return &s_renderers.get(index).front()->get_short_name()[0u];
	CATCH_ALL(nullptr)
}

RenderDevice render_get_renderer_devices(uint32_t index, uint32_t variation) {
	TRY
	CHECK(index < s_renderers.size(), "renderer index out of bounds", RenderDevice::DEVICE_NONE);
	CHECK(variation < s_renderers.get(index).size(), "renderer index out of bounds", RenderDevice::DEVICE_NONE);
	const renderer::IRenderer& renderer = *s_renderers.get(index)[variation];
	return static_cast<RenderDevice>((renderer.uses_device(Device::CPU) ? RenderDevice::DEVICE_CPU : 0)
									 | (renderer.uses_device(Device::CUDA) ? RenderDevice::DEVICE_CUDA : 0)
									 | (renderer.uses_device(Device::OPENGL) ? RenderDevice::DEVICE_OPENGL : 0));
	CATCH_ALL(RenderDevice::DEVICE_NONE)
}

Boolean render_enable_renderer(uint32_t index, uint32_t variation) {
	TRY
	CHECK(index < s_renderers.size(), "renderer index out of bounds", false);
	CHECK(variation < s_renderers.get(index).size(), "renderer index out of bounds", false);
	auto lock = std::scoped_lock(s_iterationMutex);
	s_currentRenderer = s_renderers.get(index)[variation].get();
	if(s_world.get_current_scenario() != nullptr) {
		s_currentRenderer->load_scene(s_world.get_current_scene());
		s_imageOutput = s_currentRenderer->create_output_handler(s_world.get_current_scenario()->get_resolution().x,
																 s_world.get_current_scenario()->get_resolution().y);
	} else {
		// Placeholder output handler
		s_imageOutput = s_currentRenderer->create_output_handler(1, 1);
	}
	s_currentRenderer->on_renderer_enable();

	s_targetsToEnable.clear();
	s_targetsToDisable.clear();
	return true;
	CATCH_ALL(false)
}

Boolean render_iterate(ProcessTime* iterateTime, ProcessTime* preTime, ProcessTime* postTime) {
	TRY
	auto lock = std::scoped_lock(s_iterationMutex);
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] No renderer is currently set");
		return false;
	}
	if(s_imageOutput == nullptr) {
		logError("[", FUNCTION_NAME, "] No rendertarget is currently set");
		return false;
	}

	// First update the requested render targets
	for(const auto& target : s_targetsToDisable)
		s_imageOutput->disable_render_target(target.first, target.second);
	for(const auto& target : s_targetsToEnable)
		s_imageOutput->enable_render_target(target.first, target.second);

	if(!(s_targetsToDisable.empty() && s_targetsToEnable.empty()))
		s_currentRenderer->on_render_target_changed();

	s_targetsToEnable.clear();
	s_targetsToDisable.clear();

	// Check if the scene needed a reload -> reset
	if(!s_world.is_current_scene_valid()) {
		if(s_world.get_current_scenario() == nullptr) {
			logError("[", FUNCTION_NAME, "] Failed to load scenario");
			return false;
		}
		SceneHandle hdl = s_world.load_scene(static_cast<ScenarioHandle>(s_world.get_current_scenario()), s_currentRenderer);
		ei::IVec2 res = s_world.get_current_scenario()->get_resolution();
		if(s_currentRenderer != nullptr) {
			s_currentRenderer->load_scene(hdl);
			s_imageOutput->resize(res.x, res.y);
		}
		s_screenTexture = nullptr;
	} else {
		s_world.reload_scene(s_currentRenderer);
	}
	if(!s_currentRenderer->has_scene()) {
		logError("[", FUNCTION_NAME, "] Scene not yet set for renderer");
		return false;
	}
	if(preTime != nullptr) {
		preTime->cycles = CpuProfileState::get_cpu_cycle();
		preTime->microseconds = CpuProfileState::get_process_time().count();
	}
	s_currentRenderer->pre_iteration(*s_imageOutput);
	if(preTime != nullptr) {
		preTime->cycles = CpuProfileState::get_cpu_cycle() - preTime->cycles;
		preTime->microseconds = CpuProfileState::get_process_time().count() - preTime->microseconds;
	}
	if(iterateTime != nullptr) {
		iterateTime->cycles = CpuProfileState::get_cpu_cycle();
		iterateTime->microseconds = CpuProfileState::get_process_time().count();
	}
	s_currentRenderer->iterate();
	if(iterateTime != nullptr) {
		iterateTime->cycles = CpuProfileState::get_cpu_cycle() - iterateTime->cycles;
		iterateTime->microseconds = CpuProfileState::get_process_time().count() - iterateTime->microseconds;
	}
	if(postTime != nullptr) {
		postTime->cycles = CpuProfileState::get_cpu_cycle();
		postTime->microseconds = CpuProfileState::get_process_time().count();
	}
	s_currentRenderer->post_iteration(*s_imageOutput);
	Profiler::core().create_snapshot_all();
	return true;
	CATCH_ALL(false)
}

uint32_t render_get_current_iteration() {
	if(s_imageOutput == nullptr)
		return 0u;
	return static_cast<uint32_t>(s_imageOutput->get_current_iteration() + 1);
}

Boolean render_reset() {
	TRY
	auto lock = std::scoped_lock(s_iterationMutex);
	if(s_currentRenderer != nullptr)
		s_currentRenderer->on_manual_reset();
	return true;
	CATCH_ALL(false)
}

Boolean render_save_screenshot(const char* filename, const char* targetName, Boolean variance) {
	TRY
	auto lock = std::scoped_lock(s_iterationMutex);
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] No renderer is currently set");
		return false;
	}

	// Make the image a PFM by default
	fs::path fileName = std::string(filename);
	if(fileName.extension() != ".pfm")
		fileName += ".pfm";
	if(!fileName.is_absolute())
		fileName = fs::absolute(fileName);

	// Replace tags in the file name
	const auto name = replace_screenshot_filename_tags(fileName.stem().string(), targetName, variance);
	fileName = fileName.parent_path() / fs::path(name + fileName.extension().string());

	// If necessary, create the directory we want to save our image in (alternative is to not save it at all)
	fs::path directory = fileName.parent_path();
	if(!fs::exists(directory))
		if(!fs::create_directories(directory))
			logWarning("[", FUNCTION_NAME, "] Could not create screenshot directory '", directory.string(),
					   "; the screenshot possibly may not be created");

	auto data = s_imageOutput->get_data(targetName, variance);
	const int numChannels = s_imageOutput->get_num_channels(targetName);
	ei::IVec2 res{ s_imageOutput->get_width(), s_imageOutput->get_height() };

	TextureData texData{};
	texData.data = reinterpret_cast<uint8_t*>(data.get());
	texData.components = numChannels;
	texData.format = numChannels == 1 ? FORMAT_R32F : FORMAT_RGB32F;
	texData.width = res.x;
	texData.height = res.y;
	texData.sRgb = false;
	texData.layers = 1;

	for(auto& plugin : s_plugins) {
		if(plugin.is_loaded()) {
			if(plugin.can_store_format(fileName.extension().string())) {
				if(plugin.store(fileName.string(), &texData))
					break;
			}
		}
	}
	logInfo("[", FUNCTION_NAME, "] Saved screenshot '", fileName.string(), "'");

	return true;
	CATCH_ALL(false)
}

Boolean render_save_denoised_radiance(const char* filename) {
#ifdef MUFFLON_ENABLE_OPEN_DENOISE
	TRY
	auto lock = std::scoped_lock(s_iterationMutex);
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] No renderer is currently set");
		return false;
	}
	if(s_imageOutput->get_current_iteration() == 0) {
		logError("[", FUNCTION_NAME, "] At least a single iteration must have been rendered for denoising");
		return false;
	}

	std::unique_ptr<float[]> radiance = s_imageOutput->get_data("Radiance", false);
	std::unique_ptr<float[]> output = std::make_unique<float[]>(3 * s_imageOutput->get_width() * s_imageOutput->get_height());
	std::unique_ptr<float[]> normals;
	std::unique_ptr<float[]> albedo;

	oidn::DeviceRef filterDevice = oidn::newDevice();
	filterDevice.commit();

	oidn::FilterRef filter = filterDevice.newFilter("RT");
	filter.setImage("color", radiance.get(), oidn::Format::Float3, s_imageOutput->get_width(),
					s_imageOutput->get_height());
	filter.setImage("output", output.get(), oidn::Format::Float3, s_imageOutput->get_width(),
					s_imageOutput->get_height());
	// TODO: incorporate albedo/normals if present? But currently they're noisy,
	// and OIDN wants (mostly) first-hit info (I think)
	if(s_imageOutput->has_render_target("Normal") && s_imageOutput->is_render_target_enabled("Normal", false)) {
		normals = s_imageOutput->get_data("Normal", false);
		filter.setImage("normal", normals.get(), oidn::Format::Float3, s_imageOutput->get_width(),
						s_imageOutput->get_height());
		logPedantic("[", FUNCTION_NAME, "] Using normal guidance");
	}
	if(s_imageOutput->has_render_target("Albedo") && s_imageOutput->is_render_target_enabled("Albedo", false)) {
		albedo = s_imageOutput->get_data("Albedo", false);
		filter.setImage("albedo", albedo.get(), oidn::Format::Float3, s_imageOutput->get_width(),
						s_imageOutput->get_height());
		logPedantic("[", FUNCTION_NAME, "] Using albedo guidance");
	}

	filter.set("hdr", true);
	filter.commit();
	filter.execute();
	const char* errorMsg = nullptr;
	if(filterDevice.getError(errorMsg) != oidn::Error::None) {
		logError("[", FUNCTION_NAME, "] ", errorMsg);
		return false;
	}

	// Make the image a PFM by default
	fs::path fileName = std::string(filename);
	if(fileName.extension() != ".pfm")
		fileName += ".pfm";
	if(!fileName.is_absolute())
		fileName = fs::absolute(fileName);

	const auto name = replace_screenshot_filename_tags(fileName.stem().string(), "Radiance(denoised)", false);
	fileName = fileName.parent_path() / fs::path(name + fileName.extension().string());

	// If necessary, create the directory we want to save our image in (alternative is to not save it at all)
	fs::path directory = fileName.parent_path();
	if(!fs::exists(directory))
		if(!fs::create_directories(directory))
			logWarning("[", FUNCTION_NAME, "] Could not create screenshot directory '", directory.string(),
					   "; the screenshot possibly may not be created");

	const int numChannels = s_imageOutput->get_num_channels("Radiance");
	TextureData texData{};
	texData.data = reinterpret_cast<uint8_t*>(output.get());
	texData.components = numChannels;
	texData.format = numChannels == 1 ? FORMAT_R32F : FORMAT_RGB32F;
	texData.width = s_imageOutput->get_width();
	texData.height = s_imageOutput->get_height();
	texData.sRgb = false;
	texData.layers = 1;

	for(auto& plugin : s_plugins) {
		if(plugin.is_loaded()) {
			if(plugin.can_store_format(fileName.extension().string())) {
				if(plugin.store(fileName.string(), &texData))
					break;
			}
		}
	}

	logInfo("[", FUNCTION_NAME, "] Finished denoising '", fileName.string(), "'");

	return true;
	CATCH_ALL(false)
#else // MUFFLON_ENABLE_OPEN_DENOISE
	(void)filename;
	return false;
#endif // MUFFLON_ENABLE_OPEN_DENOISE
}

uint32_t render_get_render_target_count() {
	if(s_imageOutput == nullptr)
		return 0u;
	return static_cast<uint32_t>(s_imageOutput->get_render_target_count());
}

const char* render_get_render_target_name(uint32_t index) {
	if(s_imageOutput == nullptr)
		return "";
	return s_imageOutput->get_render_target_name(index).data();
}

Boolean render_enable_render_target(const char* target, Boolean variance) {
	TRY
	const std::string targetName = target;
	s_targetsToDisable.erase({ targetName, variance });
	s_targetsToEnable.insert({ targetName, variance });
	return true;
	CATCH_ALL(false)
}

Boolean render_disable_render_target(const char* target, Boolean variance) {
	TRY
	const std::string targetName = target;
	s_targetsToEnable.erase({ targetName, variance });
	s_targetsToDisable.insert({ targetName, variance });
	return true;
	CATCH_ALL(false)
}

Boolean render_is_render_target_enabled(const char* name, Boolean variance) {
	TRY
	if(s_imageOutput == nullptr) {
		logError("[", FUNCTION_NAME, "] No renderer is enabled yet");
		return false;
	}
	const std::string targetName = name;
	// Check if we're planning on enabling/disabling first
	return (s_targetsToEnable.find({ std::string(name), variance }) != s_targetsToEnable.cend())
		|| ((s_targetsToDisable.find({ std::string(name), variance }) == s_targetsToDisable.cend())
			&& s_imageOutput->is_render_target_enabled(name, variance));
	CATCH_ALL(false)
}

Boolean render_is_render_target_required(const char* name, Boolean variance) {
	TRY
	if(s_imageOutput == nullptr) {
		logError("[", FUNCTION_NAME, "] No renderer is enabled yet");
		return false;
	}
	return s_imageOutput->is_render_target_required(name, variance);
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
	auto lock = std::scoped_lock(s_iterationMutex);
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	s_currentRenderer->get_parameters().set_param(name, value);
	s_currentRenderer->on_renderer_parameter_changed();
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
	auto lock = std::scoped_lock(s_iterationMutex);
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	s_currentRenderer->get_parameters().set_param(name, value);
	s_currentRenderer->on_renderer_parameter_changed();
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
	auto lock = std::scoped_lock(s_iterationMutex);
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	s_currentRenderer->get_parameters().set_param(name, bool(value));
	s_currentRenderer->on_renderer_parameter_changed();
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

Boolean renderer_set_parameter_enum(const char* name, int value) {
	TRY
	auto lock = std::scoped_lock(s_iterationMutex);
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	s_currentRenderer->get_parameters().set_param_enum(name, int(value));
	s_currentRenderer->on_renderer_parameter_changed();
	return true;
	CATCH_ALL(false)
}

Boolean renderer_get_parameter_enum(const char* name, int* value) {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	*value = s_currentRenderer->get_parameters().get_param_enum(name);
	return true;
	CATCH_ALL(false)
}

Boolean renderer_get_parameter_enum_count(const char* param, uint32_t* count) {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	*count = s_currentRenderer->get_parameters().get_enum_value_count(param);
	return true;
	CATCH_ALL(false)
}

Boolean renderer_get_parameter_enum_value_from_index(const char* param, uint32_t index, int* value) {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	*value = s_currentRenderer->get_parameters().get_enum_value(param, index);
	return true;
	CATCH_ALL(false)
}

Boolean renderer_get_parameter_enum_value_from_name(const char* param, const char* valueName, int* value) {
	TRY
		if(s_currentRenderer == nullptr) {
			logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
			return false;
		}
	*value = s_currentRenderer->get_parameters().get_enum_name_value(param, std::string(valueName));
	return true;
	CATCH_ALL(false)
}

Boolean renderer_get_parameter_enum_index_from_value(const char* param, int value, uint32_t* index) {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	*index = s_currentRenderer->get_parameters().get_enum_index(param, value);
	return true;
	CATCH_ALL(false)
}

Boolean renderer_get_parameter_enum_name(const char* param, int value, const char** name) {
	TRY
	if(s_currentRenderer == nullptr) {
		logError("[", FUNCTION_NAME, "] Currently, no renderer is set.");
		return false;
	}
	*name = s_currentRenderer->get_parameters().get_enum_value_name(param, value).c_str();
	return true;
	CATCH_ALL(false)
}

void profiling_enable() {
	TRY
	Profiler::core().set_enabled(true);
	CATCH_ALL(;)
}

void profiling_disable() {
	TRY
	Profiler::core().set_enabled(false);
	CATCH_ALL(;)
}

Boolean profiling_set_level(ProfilingLevel level) {
	TRY
	switch(level) {
		case ProfilingLevel::PROFILING_OFF:
			Profiler::core().set_enabled(false);
			return true;
		case ProfilingLevel::PROFILING_LOW:
			Profiler::core().set_profile_level(ProfileLevel::LOW);
			return true;
		case ProfilingLevel::PROFILING_HIGH:
			Profiler::core().set_profile_level(ProfileLevel::HIGH);
			return true;
		case ProfilingLevel::PROFILING_ALL:
			Profiler::core().set_profile_level(ProfileLevel::ALL);
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
	Profiler::core().save_current_state(path);
	return true;
	CATCH_ALL(false)
}

Boolean profiling_save_snapshots(const char* path) {
	TRY
	CHECK_NULLPTR(path, "file path", false);
	Profiler::core().save_snapshots(path);
	return true;
	CATCH_ALL(false)
}

Boolean profiling_save_total_and_snapshots(const char* path) {
	TRY
	CHECK_NULLPTR(path, "file path", false);
	Profiler::core().save_total_and_snapshots(path);
	return true;
	CATCH_ALL(false)
}

const char* profiling_get_current_state() {
	TRY
	static thread_local std::string str;
	str = Profiler::core().save_current_state();
	return str.c_str();
	CATCH_ALL(nullptr)
}

const char* profiling_get_snapshots() {
	TRY
	static thread_local std::string str;
	str = Profiler::core().save_snapshots();
	return str.c_str();
	CATCH_ALL(nullptr)
}

const char* profiling_get_total() {
	TRY
	static thread_local std::string str;
	str = Profiler::core().save_total();
	return str.c_str();
	CATCH_ALL(nullptr)
}

const char* profiling_get_total_and_snapshots() {
	TRY
	static thread_local std::string str;
	str = Profiler::core().save_total_and_snapshots();
	return str.c_str();
	CATCH_ALL(nullptr)
}

void profiling_reset() {
	TRY
	Profiler::core().reset_all();
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

Boolean mufflon_set_logger(void(*logCallback)(const char*, int)) {
	TRY
	if(s_logCallback == nullptr) {
		registerMessageHandler(delegateLog);
		disableStdHandler();

		s_logCallback = logCallback;
		// Give the new logger a status report and set the plugin loggers
		for(auto& plugin : s_plugins) {
			logInfo("[", FUNCTION_NAME, "] Loaded texture plugin '",
					plugin.get_path().string(), "'");
		}
		int count = 0;
		cuda::check_error(cudaGetDeviceCount(&count));
		if(count > 0) {
			if(s_cudaDevIndex < 0) {
				logWarning("[", FUNCTION_NAME, "] Found CUDA device(s), but none supports unified addressing; "
						   "continuing without CUDA");
			} else {
				cudaDeviceProp deviceProp;
				cuda::check_error(cudaGetDeviceProperties(&deviceProp, s_cudaDevIndex));
				logInfo("[", FUNCTION_NAME, "] Found ", count, " CUDA-capable "
						"devices; initializing device ", s_cudaDevIndex, " (", deviceProp.name, ")");
			}
		} else {
			logInfo("[", FUNCTION_NAME, "] No CUDA device found; continuing without CUDA");
		}
	}
	s_logCallback = logCallback;
	return true;
	CATCH_ALL(false)
}

Boolean mufflon_initialize() {
	TRY
	// Only once per process do we register/unregister the message handler
	static bool initialized = false;
	if(!initialized) {
		// Open the log file
		s_logFile = std::ofstream("log.txt", std::ios_base::trunc);
		// Load plugins from the DLLs directory
		s_plugins.clear();
		fs::path dllPath{};
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
#ifdef _WIN32
				if(!fs::is_directory(path) && path.extension() == ".dll") {
#else // _WIN32
				if(!fs::is_directory(path) && path.extension() == ".so") {
#endif // _WIN32
					TextureLoaderPlugin plugin{ path };
					// If we succeeded in loading (and thus have the necessary functions),
					// add it as a usable plugin
					if(plugin.is_loaded()) {
						logInfo("[", FUNCTION_NAME, "] Loaded texture plugin '",
								plugin.get_path().string(), "'");
						s_plugins.push_back(std::move(plugin));
					}
				}
			}
		}

		// Set the CUDA device to initialize the context
		int count = 0;
		const auto res = cudaGetDeviceCount(&count);
		if(res == cudaSuccess && count > 0) {
			// Parse list of CCs to determine eligible GPUs
#ifdef MUFFLON_CUDA_ARCHES
			const std::string archString{ MUFFLON_CUDA_ARCHES };
#else // MUFFLON_CUDA_ARCHES
			const std::string archString{};
#endif // MUFFLON_CUDA_ARCHES
			std::istringstream ss{ archString };
			using StringIterator = std::istream_iterator<std::string>;
			const std::vector<std::string> arches{ StringIterator{ss}, StringIterator{} };
			int minMajorCC = std::numeric_limits<int>::max();
			int minMinorCC = std::numeric_limits<int>::max();
			for(const auto& arch : arches) {
				if(const auto pos = arch.find('.'); pos != arch.npos) {
					char* end = nullptr;
					const auto major = std::strtol(arch.c_str(), &end, 10);
					if(major < minMajorCC) {
						minMajorCC = major;
						minMinorCC = std::strtol(arch.c_str() + pos + 1u, &end, 10);
					}
				}
			}
			if(minMajorCC == std::numeric_limits<int>::max()) {
				minMajorCC = 0;
				minMinorCC = 0;
			}

			// We select the device with the highest compute capability
			int devIndex = -1;
			int major = -1;
			int minor = -1;

			cudaDeviceProp deviceProp;
			for (int c = 0; c < count; ++c) {
				cuda::check_error(cudaGetDeviceProperties(&deviceProp, c));
				if(deviceProp.unifiedAddressing && deviceProp.major >= minMajorCC && deviceProp.minor >= minMinorCC) {
					if(deviceProp.major > major ||
						((deviceProp.major == major) && (deviceProp.minor > minor))) {
						major = deviceProp.major;
						minor = deviceProp.minor;
						devIndex = c;
					}
				}
			}
			if(devIndex < 0) {
				logWarning("[", FUNCTION_NAME, "] Found CUDA device(s), but none support unified addressing or have the required compute capability; "
						 "continuing without CUDA");
				mufflon::g_hasCudaEnabled = false;
			} else {
				cuda::check_error(cudaSetDevice(devIndex));
				cuda::check_error(cudaGetDeviceProperties(&deviceProp, devIndex));
				s_cudaDevIndex = devIndex;
				logInfo("[", FUNCTION_NAME, "] Found ", count, " CUDA-capable "
						"devices; initializing device ", devIndex, " (", deviceProp.name, ", compute capability ",
						deviceProp.major, ".", deviceProp.minor, ")");
			}
		} else {
			logInfo("[", FUNCTION_NAME, "] No CUDA device found; continuing without CUDA");
			mufflon::g_hasCudaEnabled = false;
		}

		// Initialize renderers
		init_renderers<false>();

		initialized = true;
	}
	return initialized;
	CATCH_ALL(false)
}

Boolean mufflon_initialize_opengl() {
	TRY
	static bool initialized = false;

	if(!initialized) {
		if(!gladLoadGL()) {
			logError("[", FUNCTION_NAME, "] gladLoadGL failed; continuing without OpenGL support");
			return false;
		}

		// Check if we satisfy our minimal version requirements
		// TODO: this should really go into 'core' at some point since we don't use the displaying capabilities anymore!
		if(GLVersion.major < 4 || (GLVersion.major == 4 && GLVersion.minor < 6)) {
			logWarning("[", FUNCTION_NAME, "] Insufficient OpenGL version (found ", GLVersion.major, ".", GLVersion.minor,
						", required 4.6); continuing without OpenGL support");
			return false;
		}

		glDebugMessageCallback(opengl_callback, nullptr);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

		logInfo("[", FUNCTION_NAME, "] Initialized OpenGL context (version ", GLVersion.major, ".", GLVersion.minor, ")");

        // initialize remaining opengl renderer
		init_renderers<true>();

		initialized = true;
	}
	return true;
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

void mufflon_destroy_opengl() {
    // destroy opengl renderers
    TRY
	for(std::size_t i = 0u; i < s_renderers.size(); ++i)
		for(auto& renderer : s_renderers.get(i))
			if (renderer->uses_device(Device::OPENGL))
				renderer.reset();
	CATCH_ALL(;)
}


void mufflon_destroy() {
	TRY
	s_imageOutput.reset();
	s_renderers.clear();
	s_plugins.clear();
	s_screenTexture.reset();
	WorldContainer::clear_instance();
	cudaDeviceReset();
	CATCH_ALL(;)
}

/*const char* get_teststring() {
	static const char* test = u8"müfflon ηρα Φ∞∡∧";
	return test;
}*/
