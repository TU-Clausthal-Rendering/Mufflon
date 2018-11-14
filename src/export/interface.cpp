#include "interface.h"
#include "util/log.hpp"
#include "util/byte_io.hpp"
#include "util/punning.hpp"
#include "util/types.hpp"
#include "ei/vector.hpp"
#include "core/cameras/pinhole.hpp"
#include "core/scene/object.hpp"
#include "core/scene/world_container.hpp"
#include "core/scene/geometry/polygon.hpp"
#include "core/scene/geometry/sphere.hpp"
#include "core/scene/lights/lights.hpp"
#include "core/scene/materials/lambert.hpp"
#include <cuda_runtime.h>
#include <type_traits>

using namespace mufflon;
using namespace mufflon::scene;
using namespace mufflon::scene::geometry;

// Helper macros for error checking and logging
#define FUNCTION_NAME __FUNCTION__
#define CHECK(x, name, retval)													\
	do {																		\
		if(x) {																	\
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

constexpr PolygonAttributeHandle INVALID_POLY_VATTR_HANDLE{
	INVALID_INDEX, INVALID_INDEX, AttributeType::ATTR_COUNT, false
};
constexpr PolygonAttributeHandle INVALID_POLY_FATTR_HANDLE{
	INVALID_INDEX, INVALID_INDEX, AttributeType::ATTR_COUNT, true
};
constexpr SphereAttributeHandle INVALID_SPHERE_ATTR_HANDLE{
	INVALID_INDEX, AttributeType::ATTR_COUNT
};

} // namespace

// Dummy type, passing another type into a lambda without needing
// to instantiate non-zero-sized data
template < class T >
struct TypeHolder {
	using Type = T;
};

template < class L1, class L2 >
inline auto switchAttributeType(AttributeType type, L1&& regular, L2&& noMatch) {
	switch(type) {
		// TODO: why not declval?
		case AttributeType::ATTR_INT:
			return regular(TypeHolder<int32_t>{});
		case AttributeType::ATTR_IVEC2:
			return regular(TypeHolder<ei::IVec2>{});
		case AttributeType::ATTR_IVEC3:
			return regular(TypeHolder<ei::IVec3>{});
		case AttributeType::ATTR_IVEC4:
			return regular(TypeHolder<ei::IVec4>{});
		case AttributeType::ATTR_UINT:
			return regular(TypeHolder<uint32_t>{});
		case AttributeType::ATTR_UVEC2:
			return regular(TypeHolder<ei::UVec2>{});
		case AttributeType::ATTR_UVEC3:
			return regular(TypeHolder<ei::UVec3>{});
		case AttributeType::ATTR_UVEC4:
			return regular(TypeHolder<ei::UVec4>{});
		case AttributeType::ATTR_FLOAT:
			return regular(TypeHolder<float>{});
		case AttributeType::ATTR_VEC2:
			return regular(TypeHolder<ei::Vec2>{});
		case AttributeType::ATTR_VEC3:
			return regular(TypeHolder<ei::Vec3>{});
		case AttributeType::ATTR_VEC4:
			return regular(TypeHolder<ei::Vec4>{});
		default:
			return noMatch();
	}
}

template < class AttrHdl >
inline AttrHdl convert_poly_to_attr(const PolygonAttributeHandle& hdl) {
	using OmAttrHandle = typename AttrHdl::OmAttrHandle;
	using CustomAttrHandle = typename AttrHdl::CustomAttrHandle;

	return AttrHdl{
		OmAttrHandle{static_cast<int>(hdl.openMeshIndex)},
		CustomAttrHandle{static_cast<size_t>(hdl.customIndex)}
	};
}

bool polygon_resize(ObjectHdl obj, size_t vertices, size_t edges, size_t faces) {
	CHECK_NULLPTR(obj, "object handle", false);
	static_cast<Object*>(obj)->resize<Polygons>(vertices, edges, faces);
	return true;
}

PolygonAttributeHandle polygon_request_vertex_attribute(ObjectHdl obj, const char* name,
														AttributeType type) {
	CHECK_NULLPTR(obj, "object handle", INVALID_POLY_VATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_POLY_VATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [name, type, &object](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		auto attr = object.request<Polygons, PolyVAttr<Type>>(name);
		return PolygonAttributeHandle{
			attr.omHandle.idx(),
			static_cast<int32_t>(attr.customHandle.index()),
			type, false
		};
	}, [type](){
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return INVALID_POLY_VATTR_HANDLE;
	});
}

PolygonAttributeHandle polygon_request_face_attribute(ObjectHdl obj,
													  const char* name,
													  AttributeType type) {
	CHECK_NULLPTR(obj, "object handle", INVALID_POLY_FATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_POLY_FATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [name, type, &object](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		auto attr = object.request<Polygons, PolyFAttr<Type>>(name);
		return PolygonAttributeHandle{
			attr.omHandle.idx(),
			static_cast<int32_t>(attr.customHandle.index()),
			type, true
		};
	}, [type]() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return INVALID_POLY_FATTR_HANDLE;
	});
}

bool polygon_remove_vertex_attribute(ObjectHdl obj, const PolygonAttributeHandle* hdl) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(hdl, "attribute", false);
	CHECK_GEQ_ZERO(hdl->openMeshIndex, "attribute index (OpenMesh)", false);
	CHECK_GEQ_ZERO(hdl->customIndex, "attribute index (custom)", false);
	CHECK(!hdl->face, "face attribute in vertex function", false);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(hdl->type, [hdl, &object](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		PolyVAttr<Type> attr = convert_poly_to_attr<PolyVAttr<Type>>(*hdl);
		object.remove<Polygons>(attr);
		return true;
	}, []() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return false;
	});
}

bool polygon_remove_face_attribute(ObjectHdl obj, const PolygonAttributeHandle* hdl) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(hdl, "attribute", false);
	CHECK_GEQ_ZERO(hdl->openMeshIndex, "attribute index (OpenMesh)", false);
	CHECK_GEQ_ZERO(hdl->customIndex, "attribute index (custom)", false);
	CHECK(hdl->face, "vertex attribute in face function", false);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(hdl->type, [hdl, &object](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		PolyFAttr<Type> attr = convert_poly_to_attr<PolyFAttr<Type>>(*hdl);
		object.remove<Polygons>(attr);
		return true;
	}, []() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return false;
	});
}

PolygonAttributeHandle polygon_find_vertex_attribute(ObjectHdl obj,
													 const char* name,
													 AttributeType type) {
	CHECK_NULLPTR(obj, "object handle", INVALID_POLY_VATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_POLY_VATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [&object, type, name](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		std::optional<PolyVAttr<Type>> attr = object.find<Polygons, PolyVAttr<Type>>(name);
		if(attr.has_value()) {
			return PolygonAttributeHandle{
				attr.value().omHandle.idx(),
				static_cast<int32_t>(attr.value().customHandle.index()),
				type, false
			};
		}
		return INVALID_POLY_VATTR_HANDLE;
	}, [type]() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return INVALID_POLY_VATTR_HANDLE;
	});
}

PolygonAttributeHandle polygon_find_face_attribute(ObjectHdl obj, const char* name,
												   AttributeType type) {
	CHECK_NULLPTR(obj, "object handle", INVALID_POLY_FATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_POLY_FATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [&object, type, name](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		std::optional<PolyFAttr<Type>> attr = object.find<Polygons, PolyFAttr<Type>>(name);
		if(attr.has_value()) {
			return PolygonAttributeHandle{
				attr.value().omHandle.idx(),
				static_cast<int32_t>(attr.value().customHandle.index()),
				type, true
			};
		}
		return INVALID_POLY_FATTR_HANDLE;
	}, [type]() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return INVALID_POLY_FATTR_HANDLE;
	});
}

VertexHdl polygon_add_vertex(ObjectHdl obj, Vec3 point, Vec3 normal, Vec2 uv) {
	CHECK_NULLPTR(obj, "object handle", VertexHdl{ INVALID_INDEX });
	PolyVHdl hdl = static_cast<Object*>(obj)->add<Polygons>(
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

	PolyFHdl hdl = static_cast<Object*>(obj)->add<Polygons>(
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
	PolyFHdl hdl = static_cast<Object*>(obj)->add<Polygons>(
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
	PolyFHdl hdl = static_cast<Object*>(obj)->add<Polygons>(
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
	PolyFHdl hdl = static_cast<Object*>(obj)->add<Polygons>(
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

	Polygons::VertexBulkReturn info = object.add_bulk<Polygons>(count, pointReader,
																normalReader, uvReader);
	if(pointsRead != nullptr)
		*pointsRead = info.readPoints;
	if(pointsRead != nullptr)
		*normalsRead = info.readNormals;
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
	Polygons::VertexBulkReturn info = object.add_bulk<Polygons>(count, pointReader,
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

bool polygon_set_vertex_attribute(ObjectHdl obj, const PolygonAttributeHandle* attr,
								  VertexHdl vertex, AttributeType type,
								  void* value) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(attr, "attribute handle", false);
	CHECK_NULLPTR(value, "attribute value", false);
	CHECK(!attr->face, "Face attribute in vertex function", false);
	CHECK_GEQ_ZERO(vertex, "vertex index", false);
	CHECK_GEQ_ZERO(attr->openMeshIndex, "attribute index (OpenMesh)", false);
	CHECK_GEQ_ZERO(attr->customIndex, "attribute index (custom)", false);
	Object& object = *static_cast<Object*>(obj);
	if(vertex >= static_cast<int>(object.get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 vertex, " >= ", object.get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return false;
	}

	return switchAttributeType(type, [&object, attr, vertex, value](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		auto attribute = object.aquire<Polygons>(convert_poly_to_attr<PolyVAttr<Type>>(*attr));
		(*attribute.aquire<Device::CPU>())[vertex] = *static_cast<Type*>(value);
		return true;
	}, []() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return false;
	});
}

bool polygon_set_face_attribute(ObjectHdl obj, const PolygonAttributeHandle* attr,
								FaceHdl face, AttributeType type, void* value) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(attr, "attribute handle", false);
	CHECK_NULLPTR(value, "attribute value", false);
	CHECK(attr->face, "Vertex attribute in face function", false);
	CHECK_GEQ_ZERO(face, "face index", false);
	CHECK_GEQ_ZERO(attr->openMeshIndex, "attribute index (OpenMesh)", false);
	CHECK_GEQ_ZERO(attr->customIndex, "attribute index (custom)", false);
	Object& object = *static_cast<Object*>(obj);
	if(face >= static_cast<int>(object.get_geometry<Polygons>().get_face_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 face, " >= ", object.get_geometry<Polygons>().get_face_count(),
				 ")");
		return false;
	}

	return switchAttributeType(type, [&object, attr, face, value](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		auto attribute = object.aquire<Polygons>(convert_poly_to_attr<PolyFAttr<Type>>(*attr));
		(*attribute.aquire<Device::CPU>())[face] = *static_cast<Type*>(value);
		return true;
	}, []() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return false;
	});
}

bool polygon_set_material_idx(ObjectHdl obj, FaceHdl face, MatIdx idx) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_GEQ_ZERO(face, "face index", false);
	Object& object = *static_cast<Object*>(obj);
	if(face >= static_cast<int>(object.get_geometry<Polygons>().get_face_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 face, " >= ", object.get_geometry<Polygons>().get_face_count(),
				 ")");
		return false;
	}

	(*object.get_mat_indices<Polygons>().aquire())[face] = idx;
	return true;
}

size_t polygon_set_vertex_attribute_bulk(ObjectHdl obj, const PolygonAttributeHandle* attr,
										 VertexHdl startVertex, AttributeType type,
										 size_t count, FILE* stream) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	CHECK_NULLPTR(attr, "attribute handle", INVALID_SIZE);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK(!attr->face, "Face attribute in vertex function", false);
	CHECK_GEQ_ZERO(startVertex, "start vertex index", INVALID_SIZE);
	CHECK_GEQ_ZERO(attr->openMeshIndex, "attribute index (OpenMesh)", INVALID_SIZE);
	CHECK_GEQ_ZERO(attr->customIndex, "attribute index (custom)", INVALID_SIZE);
	Object& object = *static_cast<Object*>(obj);
	if(startVertex >= static_cast<int>(object.get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 startVertex, " >= ", object.get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return INVALID_SIZE;
	}
	util::FileReader attrStream{ stream };

	return switchAttributeType(type, [&object, attr, startVertex, count, &attrStream](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		return object.add_bulk<Polygons>(convert_poly_to_attr<PolyVAttr<Type>>(*attr),
										 PolyVHdl{ static_cast<int>(startVertex) },
										 count, attrStream);
	}, []() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return INVALID_SIZE;
	});
}

size_t polygon_set_face_attribute_bulk(ObjectHdl obj, const PolygonAttributeHandle* attr,
									   FaceHdl startFace, AttributeType type,
									   size_t count, FILE* stream) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	CHECK_NULLPTR(attr, "attribute handle", INVALID_SIZE);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK(attr->face, "Vertex attribute in face function", false);
	CHECK_GEQ_ZERO(startFace, "start face index", INVALID_SIZE);
	CHECK_GEQ_ZERO(attr->openMeshIndex, "attribute index (OpenMesh)", INVALID_SIZE);
	CHECK_GEQ_ZERO(attr->customIndex, "attribute index (custom)", INVALID_SIZE);
	Object& object = *static_cast<Object*>(obj);
	if(startFace >= static_cast<int>(object.get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 startFace, " >= ", object.get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return INVALID_SIZE;
	}
	util::FileReader attrStream{ stream };

	return switchAttributeType(type, [&object, attr, startFace, count, &attrStream](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		return object.add_bulk<Polygons>(convert_poly_to_attr<PolyFAttr<Type>>(*attr),
										 PolyFHdl{ static_cast<int>(startFace) },
										 count, attrStream);
	}, []() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return INVALID_SIZE;
	});
}

size_t polygon_set_material_idx_bulk(ObjectHdl obj, FaceHdl startFace, size_t count,
									 FILE* stream) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK_GEQ_ZERO(startFace, "start face index", INVALID_SIZE);
	Object& object = *static_cast<Object*>(obj);
	if(startFace >= static_cast<int>(object.get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 startFace, " >= ", object.get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return INVALID_SIZE;
	}
	util::FileReader matStream{ stream };

	return object.add_bulk<Polygons>(object.get_mat_indices<Polygons>(),
									 PolyFHdl{ static_cast<int>(startFace) },
									 count, matStream);
}

size_t polygon_get_vertex_count(ObjectHdl obj) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.get_geometry<Polygons>().get_vertex_count();
}

size_t polygon_get_edge_count(ObjectHdl obj) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.get_geometry<Polygons>().get_edge_count();
}

size_t polygon_get_face_count(ObjectHdl obj) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.get_geometry<Polygons>().get_face_count();
}

size_t polygon_get_triangle_count(ObjectHdl obj) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.get_geometry<Polygons>().get_triangle_count();
}

size_t polygon_get_quad_count(ObjectHdl obj) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.get_geometry<Polygons>().get_quad_count();
}

bool polygon_get_bounding_box(ObjectHdl obj, Vec3* min, Vec3* max) {
	CHECK_NULLPTR(obj, "object handle", false);
	const Object& object = *static_cast<const Object*>(obj);
	const ei::Box& aabb = object.get_geometry<Polygons>().get_bounding_box();
	if(min != nullptr)
		*min = util::pun<Vec3>(aabb.min);
	if(max != nullptr)
		*max = util::pun<Vec3>(aabb.max);
	return true;
}

bool spheres_resize(ObjectHdl obj, size_t count) {
	CHECK_NULLPTR(obj, "object handle", false);
	static_cast<Object*>(obj)->resize<Spheres>(count);
	return true;
}

SphereAttributeHandle spheres_request_attribute(ObjectHdl obj, const char* name,
												AttributeType type) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SPHERE_ATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_SPHERE_ATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [name, type, &object](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		return SphereAttributeHandle{
			static_cast<int>(object.request<Spheres, Type>(name).index()),
			type
		};
	}, [type]() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return INVALID_SPHERE_ATTR_HANDLE;
	});
}

bool spheres_remove_attribute(ObjectHdl obj, const SphereAttributeHandle* hdl) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(hdl, "attribute", false);
	CHECK_GEQ_ZERO(hdl->index, "attribute index", false);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(hdl->type, [hdl, &object](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		SphereAttr<Type> attr{ static_cast<size_t>(hdl->index) };
		object.remove<Spheres>(attr);
		return true;
	}, []() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return false;
	});
}

SphereAttributeHandle spheres_find_attribute(ObjectHdl obj, const char* name,
											 AttributeType type) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SPHERE_ATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_SPHERE_ATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [&object, type, name](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		std::optional<SphereAttr<Type>> attr = object.find<Spheres, Type>(name);
		if(attr.has_value()) {
			return SphereAttributeHandle{
				static_cast<IndexType>(attr.value().index()),
				type
			};
		}
		return INVALID_SPHERE_ATTR_HANDLE;
	}, [type]() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return INVALID_SPHERE_ATTR_HANDLE;
	});
}

SphereHdl spheres_add_sphere(ObjectHdl obj, Vec3 point, float radius) {
	CHECK_NULLPTR(obj, "object handle", SphereHdl{ INVALID_INDEX });
	SphereVHdl hdl = static_cast<Object*>(obj)->add<Spheres>(
		util::pun<ei::Vec3>(point), radius);
	return SphereHdl{ static_cast<IndexType>(hdl) };
}

SphereHdl spheres_add_sphere_material(ObjectHdl obj, Vec3 point, float radius,
								MatIdx idx) {
	CHECK_NULLPTR(obj, "object handle", SphereHdl{ INVALID_INDEX });
	SphereVHdl hdl = static_cast<Object*>(obj)->add<Spheres>(
		util::pun<ei::Vec3>(point), radius, idx);
	return SphereHdl{ static_cast<IndexType>(hdl) };
}

SphereHdl spheres_add_sphere_bulk(ObjectHdl obj, size_t count,
									 FILE* stream, size_t* readSpheres) {
	CHECK_NULLPTR(obj, "object handle", SphereHdl{ INVALID_INDEX } );
	CHECK_NULLPTR(stream, "sphere stream descriptor", SphereHdl{ INVALID_INDEX });
	Object& object = *static_cast<Object*>(obj);
	mufflon::util::FileReader sphereReader{ stream };

	Spheres::BulkReturn info = object.add_bulk<Spheres>(count, sphereReader);
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
	Spheres::BulkReturn info = object.add_bulk<Spheres>(count, sphereReader,
														aabb);
	if(readSpheres != nullptr)
		*readSpheres = info.readSpheres;
	return SphereHdl{ static_cast<IndexType>(info.handle) };
}

bool spheres_set_attribute(ObjectHdl obj, const SphereAttributeHandle* attr,
						   SphereHdl sphere, AttributeType type, void* value) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_NULLPTR(attr, "attribute handle", false);
	CHECK_NULLPTR(value, "attribute value", false);
	CHECK_GEQ_ZERO(sphere, "sphere index", false);
	CHECK_GEQ_ZERO(attr->index, "attribute index", false);
	Object& object = *static_cast<Object*>(obj);
	if(sphere >= static_cast<int>(object.get_geometry<Spheres>().get_sphere_count())) {
		logError("[", FUNCTION_NAME, "] Sphere index out of bounds (",
				 sphere, " >= ", object.get_geometry<Spheres>().get_sphere_count(),
				 ")");
		return false;
	}

	return switchAttributeType(type, [&object, attr, sphere, value](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		SphereAttr<Type> sphereAttr{ static_cast<size_t>(attr->index) };
		auto attribute = object.aquire<Spheres>(sphereAttr);
		(*attribute.aquire<Device::CPU>())[sphere] = *static_cast<Type*>(value);
		return true;
	}, []() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return false;
	});
}

bool sphere_set_material_idx(ObjectHdl obj, SphereHdl sphere, MatIdx idx) {
	CHECK_NULLPTR(obj, "object handle", false);
	CHECK_GEQ_ZERO(sphere, "sphere index", false);
	Object& object = *static_cast<Object*>(obj);
	if(sphere >= static_cast<int>(object.get_geometry<Spheres>().get_sphere_count())) {
		logError("[", FUNCTION_NAME, "] Sphere index out of bounds (",
				 sphere, " >= ", object.get_geometry<Spheres>().get_sphere_count(),
				 ")");
		return false;
	}

	(*object.get_mat_indices<Spheres>().aquire())[sphere] = idx;
	return true;
}

size_t spheres_set_attribute_bulk(ObjectHdl obj, const SphereAttributeHandle* attr,
								  SphereHdl startSphere,
								  AttributeType type, size_t count,
								  FILE* stream) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	CHECK_NULLPTR(attr, "attribute handle", INVALID_SIZE);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK_GEQ_ZERO(startSphere, "start sphere index", INVALID_SIZE);
	CHECK_GEQ_ZERO(attr->index, "attribute index", INVALID_SIZE);
	Object& object = *static_cast<Object*>(obj);
	if(startSphere >= static_cast<int>(object.get_geometry<Spheres>().get_sphere_count())) {
		logError("[", FUNCTION_NAME, "] Sphere index out of bounds (",
				 startSphere, " >= ", object.get_geometry<Spheres>().get_sphere_count(),
				 ")");
		return INVALID_SIZE;
	}
	util::FileReader attrStream{ stream };

	return switchAttributeType(type, [&object, attr, type, startSphere, count, &attrStream](const auto& val) {
		using Type = std::decay_t<decltype(val)>::Type;
		SphereAttr<Type> sphereAttr{ static_cast<size_t>(attr->index) };
		return INVALID_SIZE;
		/*return object.add_bulk<Spheres>(sphereAttr,
										SphereHdl{ static_cast<size_t>(startSphere) },
										count, attrStream);*/
		// TODO
	}, []() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
		return INVALID_SIZE;
	});
}

size_t sphere_set_material_idx_bulk(ObjectHdl obj, SphereHdl startSphere, size_t count,
									 FILE* stream) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	CHECK_NULLPTR(stream, "attribute stream", INVALID_SIZE);
	CHECK_GEQ_ZERO(startSphere, "start sphere index", INVALID_SIZE);
	Object& object = *static_cast<Object*>(obj);
	if(startSphere >= static_cast<int>(object.get_geometry<Spheres>().get_sphere_count())) {
		logError("[", FUNCTION_NAME, "] Sphere index out of bounds (",
				 startSphere, " >= ", object.get_geometry<Spheres>().get_sphere_count(),
				 ")");
		return INVALID_SIZE;
	}
	util::FileReader matStream{ stream };

	// TODO
	/*return object.add_bulk<Spheres>(object.get_mat_indices<Spheres>(),
									SphereHdl{ static_cast<size_t>(startSphere) },
									count, matStream);*/
	return INVALID_SIZE;
}

size_t spheres_get_sphere_count(ObjectHdl obj) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SIZE);
	const Object& object = *static_cast<const Object*>(obj);
	return object.get_geometry<Spheres>().get_sphere_count();
}

bool spheres_get_bounding_box(ObjectHdl obj, Vec3* min, Vec3* max) {
	CHECK_NULLPTR(obj, "object handle", false);
	const Object& object = *static_cast<const Object*>(obj);
	const ei::Box& aabb = object.get_bounding_box<Spheres>();
	if(min != nullptr)
		*min = util::pun<Vec3>(aabb.min);
	if(max != nullptr)
		*max = util::pun<Vec3>(aabb.max);
	return true;
}

ObjectHdl world_create_object() {
	return static_cast<ObjectHdl>(WorldContainer::instance().create_object());
}

InstanceHdl world_create_instance(ObjectHdl obj) {
	CHECK_NULLPTR(obj, "object handle", nullptr);
	ObjectHandle hdl = static_cast<Object*>(obj);
	return static_cast<InstanceHdl>(WorldContainer::instance().create_instance(hdl));
}

ScenarioHdl world_create_scenario(const char* name) {
	CHECK_NULLPTR(name, "scenario name", nullptr);
	auto hdl = WorldContainer::instance().add_scenario(Scenario{ name });
	// TODO
	//return static_cast<ScenarioHdl>(hdl);
	return nullptr;
}

MaterialHdl world_add_lambert_material(const char* name) {
	CHECK_NULLPTR(name, "material name", nullptr);
	MaterialHandle hdl = WorldContainer::instance().add_material(std::make_unique<materials::Lambert>());
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Error creating lambert material");
		return nullptr;
	}
	hdl->set_name(name);
	// TODO: fill with life

	return static_cast<MaterialHandle>(hdl);
}

CameraHdl world_add_pinhole_camera(const char* name, Vec3 position, Vec3 dir,
								   Vec3 up, float near, float far, float tanVFov) {
	CHECK_NULLPTR(name, "camera name", nullptr);
	CameraHandle hdl = WorldContainer::instance().add_camera(std::make_unique<cameras::Pinhole>());
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Error creating pinhole camera");
		return nullptr;
	}
	// TODO: how about we add an actual constructor
	hdl->set_name(name);
	hdl->set_near(near);
	hdl->set_far(far);
	hdl->move(position.x, position.y, position.z);
	// TODO: set direction and up-vector

	return static_cast<CameraHdl>(hdl);
}

LightHdl world_add_point_light(const char* name, Vec3 position, Vec3 intensity) {
	CHECK_NULLPTR(name, "pointlight name", nullptr);
	auto hdl = WorldContainer::instance().add_light(name, lights::PointLight{
													util::pun<ei::Vec3>(position),
													util::pun<ei::Vec3>(intensity) });
	// TODO
	//return static_cast<LightHdl>(hdl);
	return nullptr;
}

LightHdl world_add_spot_light(const char* name, Vec3 position, Vec3 direction,
							  Vec3 intensity, float openingAngle,
							  float falloffStart) {
	CHECK_NULLPTR(name, "spotlight name", nullptr);
	ei::Vec3 dir = ei::normalize(util::pun<ei::Vec3>(direction));
	auto hdl = WorldContainer::instance().add_light(name, lights::SpotLight{
													util::pun<ei::Vec3>(position),
													ei::packOctahedral32(dir),
													util::pun<ei::Vec3>(intensity),
													__float2half(openingAngle),
													__float2half(falloffStart) });
	// TODO
	//return static_cast<LightHdl>(hdl);
	return nullptr;
}

LightHdl world_add_directional_light(const char* name, Vec3 direction,
									 Vec3 radiance) {
	CHECK_NULLPTR(name, "directional light name", nullptr);
	auto hdl = WorldContainer::instance().add_light(name, lights::DirectionalLight{
													util::pun<ei::Vec3>(direction),
													util::pun<ei::Vec3>(radiance) });
	// TODO
	//return static_cast<LightHdl>(hdl);
	return nullptr;
}

LightHdl world_add_envmap_light(const char* name, TextureHdl envmap) {
	CHECK_NULLPTR(name, "directional light name", nullptr);
	CHECK_NULLPTR(envmap, "environment map", nullptr);
	// TODO
	return nullptr;
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
			// TODO
			return nullptr;
		}
		case LightType::LIGHT_SPOT: {
			auto hdl = WorldContainer::instance().get_spot_light(name);
			// TODO
			return nullptr;
		}
		case LightType::LIGHT_DIRECTIONAL: {
			auto hdl = WorldContainer::instance().get_dir_light(name);
			// TODO
			return nullptr;
		}
		case LightType::LIGHT_ENVMAP: {
			auto hdl = WorldContainer::instance().get_env_light(name);
			// TODO
			return nullptr;
		}
		default: {
			logError("[", FUNCTION_NAME, "] Unknown light type");
			return nullptr;
		}
	}
}

SceneHdl world_load_scenario(ScenarioHdl scenario) {
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	// TODO
	/*SceneHandle hdl = WorldContainer::instance().load_scene(scenario);
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Failed to load scenario");
		return nullptr;
	}
	return static_cast<SceneHdl>(hdl);*/
	return nullptr;
}

SceneHdl world_get_current_scene() {
	return static_cast<SceneHdl>(WorldContainer::instance().get_current_scene());
}