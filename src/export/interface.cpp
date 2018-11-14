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
#define FUNCTION_NAME __func__
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
	static_cast<Object*>(obj)->template resize<Polygons>(vertices, edges, faces);
	return true;
}

PolygonAttributeHandle polygon_request_vertex_attribute(ObjectHdl obj, const char* name,
														AttributeType type) {
	CHECK_NULLPTR(obj, "object handle", INVALID_POLY_VATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_POLY_VATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [name, type, &object](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		auto attr = object.template request<Polygons, PolyVAttr<Type>>(name);
		return PolygonAttributeHandle{
			attr.omHandle.idx(),
			static_cast<int32_t>(attr.customHandle.index()),
			type, false
		};
	}, [](){
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
		using Type = typename std::decay_t<decltype(val)>::Type;
		auto attr = object.template request<Polygons, PolyFAttr<Type>>(name);
		return PolygonAttributeHandle{
			attr.omHandle.idx(),
			static_cast<int32_t>(attr.customHandle.index()),
			type, true
		};
	}, []() {
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
		using Type = typename std::decay_t<decltype(val)>::Type;
		PolyVAttr<Type> attr = convert_poly_to_attr<PolyVAttr<Type>>(*hdl);
		object.template remove<Polygons>(attr);
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
		using Type = typename std::decay_t<decltype(val)>::Type;
		PolyFAttr<Type> attr = convert_poly_to_attr<PolyFAttr<Type>>(*hdl);
		object.template remove<Polygons>(attr);
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
	}, []() {
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
	}, []() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
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
	if(vertex >= static_cast<int>(object.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 vertex, " >= ", object.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return false;
	}

	return switchAttributeType(type, [&object, attr, vertex, value](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		auto attribute = object.template aquire<Polygons>(convert_poly_to_attr<PolyVAttr<Type>>(*attr));
		(*attribute.template aquire<Device::CPU>())[vertex] = *static_cast<Type*>(value);
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
	if(face >= static_cast<int>(object.template get_geometry<Polygons>().get_face_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 face, " >= ", object.template get_geometry<Polygons>().get_face_count(),
				 ")");
		return false;
	}

	return switchAttributeType(type, [&object, attr, face, value](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		auto attribute = object.template aquire<Polygons>(convert_poly_to_attr<PolyFAttr<Type>>(*attr));
		(*attribute.template aquire<Device::CPU>())[face] = *static_cast<Type*>(value);
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
	if(startVertex >= static_cast<int>(object.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Vertex index out of bounds (",
				 startVertex, " >= ", object.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return INVALID_SIZE;
	}
	util::FileReader attrStream{ stream };

	return switchAttributeType(type, [&object, attr, startVertex, count, &attrStream](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		return object.template add_bulk<Polygons>(convert_poly_to_attr<PolyVAttr<Type>>(*attr),
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
	if(startFace >= static_cast<int>(object.template get_geometry<Polygons>().get_vertex_count())) {
		logError("[", FUNCTION_NAME, "] Face index out of bounds (",
				 startFace, " >= ", object.template get_geometry<Polygons>().get_vertex_count(),
				 ")");
		return INVALID_SIZE;
	}
	util::FileReader attrStream{ stream };

	return switchAttributeType(type, [&object, attr, startFace, count, &attrStream](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		return object.template add_bulk<Polygons>(convert_poly_to_attr<PolyFAttr<Type>>(*attr),
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

bool polygon_get_bounding_box(ObjectHdl obj, Vec3* min, Vec3* max) {
	CHECK_NULLPTR(obj, "object handle", false);
	const Object& object = *static_cast<const Object*>(obj);
	const ei::Box& aabb = object.template get_geometry<Polygons>().get_bounding_box();
	if(min != nullptr)
		*min = util::pun<Vec3>(aabb.min);
	if(max != nullptr)
		*max = util::pun<Vec3>(aabb.max);
	return true;
}

bool spheres_resize(ObjectHdl obj, size_t count) {
	CHECK_NULLPTR(obj, "object handle", false);
	static_cast<Object*>(obj)->template resize<Spheres>(count);
	return true;
}

SphereAttributeHandle spheres_request_attribute(ObjectHdl obj, const char* name,
												AttributeType type) {
	CHECK_NULLPTR(obj, "object handle", INVALID_SPHERE_ATTR_HANDLE);
	CHECK_NULLPTR(name, "attribute name", INVALID_SPHERE_ATTR_HANDLE);
	Object& object = *static_cast<Object*>(obj);

	return switchAttributeType(type, [name, type, &object](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		return SphereAttributeHandle{
			static_cast<int>(object.template request<Spheres, Type>(name).index()),
			type
		};
	}, []() {
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
		using Type = typename std::decay_t<decltype(val)>::Type;
		SphereAttr<Type> attr{ static_cast<size_t>(hdl->index) };
		object.template remove<Spheres>(attr);
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
		using Type = typename std::decay_t<decltype(val)>::Type;
		std::optional<SphereAttr<Type>> attr = object.template find<Spheres, Type>(name);
		if(attr.has_value()) {
			return SphereAttributeHandle{
				static_cast<IndexType>(attr.value().index()),
				type
			};
		}
		return INVALID_SPHERE_ATTR_HANDLE;
	}, []() {
		logError("[", FUNCTION_NAME, "] Unknown attribute type");
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

bool spheres_set_attribute(ObjectHdl obj, const SphereAttributeHandle* attr,
						   SphereHdl sphere, AttributeType type, void* value) {
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

	return switchAttributeType(type, [&object, attr, sphere, value](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		SphereAttr<Type> sphereAttr{ static_cast<size_t>(attr->index) };
		auto attribute = object.template aquire<Spheres>(sphereAttr);
		(*attribute.template aquire<Device::CPU>())[sphere] = *static_cast<Type*>(value);
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
								  SphereHdl startSphere,
								  AttributeType type, size_t count,
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

	return switchAttributeType(type, [&object, attr, startSphere, count, &attrStream](const auto& val) {
		using Type = typename std::decay_t<decltype(val)>::Type;
		SphereAttr<Type> sphereAttr{ static_cast<size_t>(attr->index) };
		return object.template add_bulk<Spheres>(sphereAttr,
										SphereVHdl{ static_cast<size_t>(startSphere) },
										count, attrStream);
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
	const auto& hdl = WorldContainer::instance().add_scenario(Scenario{ name });
	return static_cast<ScenarioHdl>(&hdl->second);
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
								   Vec3 up, float near, float far, float vFov) {
	CHECK_NULLPTR(name, "camera name", nullptr);
	CameraHandle hdl = WorldContainer::instance().add_camera(std::make_unique<cameras::Pinhole>(
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
	SceneHandle hdl = WorldContainer::instance().load_scene(*static_cast<const Scenario*>(scenario));
	if(hdl == nullptr) {
		logError("[", FUNCTION_NAME, "] Failed to load scenario");
		return nullptr;
	}
	return static_cast<SceneHdl>(hdl);
}

SceneHdl world_get_current_scene() {
	return static_cast<SceneHdl>(WorldContainer::instance().get_current_scene());
}

const char* scenario_get_name(ScenarioHdl scenario) {
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	return static_cast<const Scenario*>(scenario)->get_name().c_str();
}

size_t scenario_get_global_lod_level(ScenarioHdl scenario) {
	CHECK_NULLPTR(scenario, "scenario handle", Scenario::NO_CUSTOM_LOD);
	return static_cast<const Scenario*>(scenario)->get_global_lod_level();
}

bool scenario_set_global_lod_level(ScenarioHdl scenario, LodLevel level) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	static_cast<Scenario*>(scenario)->set_global_lod_level(level);
	return true;
}

IVec2 scenario_get_resolution(ScenarioHdl scenario) {
	CHECK_NULLPTR(scenario, "scenario handle", (IVec2{ 0, 0 }));
	return util::pun<IVec2>(static_cast<const Scenario*>(scenario)->get_resolution());
}

bool scenario_set_resolution(ScenarioHdl scenario, IVec2 res) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	static_cast<Scenario*>(scenario)->set_resolution(util::pun<ei::IVec2>(res));
	return true;
}

CameraHdl scenario_get_camera(ScenarioHdl scenario) {
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	return static_cast<CameraHdl>(static_cast<const Scenario*>(scenario)->get_camera());
}

bool scenario_set_camera(ScenarioHdl scenario, CameraHdl cam) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	static_cast<Scenario*>(scenario)->set_camera(static_cast<CameraHandle>(cam));
	return true;
}

bool scenario_is_object_masked(ScenarioHdl scenario, ObjectHdl obj) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(obj, "object handle", false);
	return static_cast<const Scenario*>(scenario)->is_masked(static_cast<const Object*>(obj));
}

bool scenario_mask_object(ScenarioHdl scenario, ObjectHdl obj) {
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

bool scenario_set_object_lod(ScenarioHdl scenario, ObjectHdl obj,
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

bool scenario_add_light(ScenarioHdl scenario, const char* name) {
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

bool scenario_remove_light_by_index(ScenarioHdl scenario, size_t index) {
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

bool scenario_remove_light_by_named(ScenarioHdl scenario, const char* name) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(name, "light name", false);
	Scenario& scen = *static_cast<Scenario*>(scenario);
	scen.remove_light(name);
	return true;
}

MatIdx scenario_declare_material_slot(ScenarioHdl scenario,
									  const char* name) {
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_MATERIAL);
	CHECK_NULLPTR(name, "material name", INVALID_MATERIAL);
	return static_cast<Scenario*>(scenario)->declare_material_slot(name);
}

MatIdx scenario_get_material_slot(ScenarioHdl scenario,
								  const char* name) {
	CHECK_NULLPTR(scenario, "scenario handle", INVALID_MATERIAL);
	CHECK_NULLPTR(name, "material name", INVALID_MATERIAL);
	return static_cast<const Scenario*>(scenario)->get_material_slot_index(name);
}

MaterialHdl scenario_get_assigned_material(ScenarioHdl scenario,
										   MatIdx index) {
	CHECK_NULLPTR(scenario, "scenario handle", nullptr);
	CHECK(index < INVALID_MATERIAL, "Invalid material index", nullptr);
	return static_cast<MaterialHdl>(static_cast<const Scenario*>(scenario)->get_assigned_material(index));
}

bool scenario_assign_material(ScenarioHdl scenario, MatIdx index,
							  MaterialHdl handle) {
	CHECK_NULLPTR(scenario, "scenario handle", false);
	CHECK_NULLPTR(handle, "material handle", false);
	CHECK(index < INVALID_MATERIAL, "Invalid material index", false);
	static_cast<Scenario*>(scenario)->assign_material(index, static_cast<MaterialHandle>(handle));
	return true;
}

bool scene_get_bounding_box(SceneHdl scene, Vec3* min, Vec3* max) {
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

bool world_get_point_light_position(LightHdl hdl, Vec3* pos) {
	CHECK_NULLPTR(hdl, "pointlight handle", false);
	const lights::PointLight& light = *static_cast<const lights::PointLight*>(hdl);
	if(pos != nullptr)
		*pos = util::pun<Vec3>(light.position);
	return true;
}

bool world_get_point_light_intensity(LightHdl hdl, Vec3* intensity) {
	CHECK_NULLPTR(hdl, "pointlight handle", false);
	const lights::PointLight& light = *static_cast<const lights::PointLight*>(hdl);
	if(intensity != nullptr)
		*intensity = util::pun<Vec3>(light.intensity);
	return true;
}

bool world_set_point_light_position(LightHdl hdl, Vec3 pos) {
	CHECK_NULLPTR(hdl, "pointlight handle", false);
	lights::PointLight& light = *static_cast<lights::PointLight*>(hdl);
	light.position = util::pun<ei::Vec3>(pos);
	return true;
}

bool world_set_point_light_intensity(LightHdl hdl, Vec3 intensity) {
	CHECK_NULLPTR(hdl, "pointlight handle", false);
	lights::PointLight& light = *static_cast<lights::PointLight*>(hdl);
	light.intensity = util::pun<ei::Vec3>(intensity);
	return true;
}

bool world_get_spot_light_position(LightHdl hdl, Vec3* pos) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	const lights::SpotLight& light = *static_cast<const lights::SpotLight*>(hdl);
	if(pos != nullptr)
		*pos = util::pun<Vec3>(light.position);
	return true;
}

bool world_get_spot_light_intensity(LightHdl hdl, Vec3* intensity) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	const lights::SpotLight& light = *static_cast<const lights::SpotLight*>(hdl);
	if(intensity != nullptr)
		*intensity = util::pun<Vec3>(light.intensity);
	return true;
}

bool world_get_spot_light_direction(LightHdl hdl, Vec3* direction) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	const lights::SpotLight& light = *static_cast<const lights::SpotLight*>(hdl);
	if(direction != nullptr)
		*direction = util::pun<Vec3>(ei::unpackOctahedral32(light.direction));
	return true;
}

bool world_get_spot_light_angle(LightHdl hdl, float* angle) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	const lights::SpotLight& light = *static_cast<const lights::SpotLight*>(hdl);
	if(angle != nullptr)
		*angle = std::acos(__half2float(light.cosThetaMax));
	return true;
}

bool world_get_spot_light_falloff(LightHdl hdl, float* falloff) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	const lights::SpotLight& light = *static_cast<const lights::SpotLight*>(hdl);
	if(falloff != nullptr)
		*falloff = std::acos(__half2float(light.cosFalloffStart));
	return true;
}

bool world_set_spot_light_position(LightHdl hdl, Vec3 pos) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	lights::SpotLight& light = *static_cast<lights::SpotLight*>(hdl);
	light.position = util::pun<ei::Vec3>(pos);
	return true;
}

bool world_set_spot_light_intensity(LightHdl hdl, Vec3 intensity) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	lights::SpotLight& light = *static_cast<lights::SpotLight*>(hdl);
	light.intensity = util::pun<ei::Vec3>(intensity);
	return true;
}

bool world_set_spot_light_direction(LightHdl hdl, Vec3 direction) {
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

bool world_set_spot_light_angle(LightHdl hdl, float angle) {
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

bool world_set_spot_light_falloff(LightHdl hdl, float falloff) {
	CHECK_NULLPTR(hdl, "spotlight handle", false);
	lights::SpotLight& light = *static_cast<lights::SpotLight*>(hdl);
	// Clamp it to the opening angle!
	float actualFalloff = std::fmod(falloff, 2.f * ei::PI);
	if(actualFalloff < 0.f)
		actualFalloff += 2.f*ei::PI;
	float cosFalloff = std::cos(actualFalloff);
	if(__half2float(light.cosThetaMax) > cosFalloff) {
		logWarning("[", FUNCTION_NAME, "] Spotlight falloff angle cannot be larger than"
				   " its opening angle");
		light.cosFalloffStart = light.cosThetaMax;
	} else {
		light.cosFalloffStart = __float2half(cosFalloff);
	}
	return true;
}

bool world_get_dir_light_direction(LightHdl hdl, Vec3* direction) {
	CHECK_NULLPTR(hdl, "directional light handle", false);
	const lights::DirectionalLight& light = *static_cast<const lights::DirectionalLight*>(hdl);
	if(direction != nullptr)
		*direction = util::pun<Vec3>(light.direction);
	return true;
}

bool world_get_dir_light_radiance(LightHdl hdl, Vec3* radiance) {
	CHECK_NULLPTR(hdl, "directional light handle", false);
	const lights::DirectionalLight& light = *static_cast<const lights::DirectionalLight*>(hdl);
	if(radiance != nullptr)
		*radiance = util::pun<Vec3>(light.radiance);
	return true;
}

bool world_set_dir_light_direction(LightHdl hdl, Vec3 direction) {
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

bool world_set_dir_light_radiance(LightHdl hdl, Vec3 radiance) {
	CHECK_NULLPTR(hdl, "directional light handle", false);
	lights::DirectionalLight& light = *static_cast<lights::DirectionalLight*>(hdl);
	light.radiance = util::pun<ei::Vec3>(radiance);
	return true;
}
