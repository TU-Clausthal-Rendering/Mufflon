#pragma once

#include "api.hpp"
#include "core/scene/geometry/polygon.hpp"

extern "C" {

#include <stdint.h>

typedef struct  {
	float x;
	float y;
} Vec2;

typedef struct {
	float x;
	float y;
	float z;
} Vec3;

typedef struct {
	unsigned int x;
	unsigned int y;
	unsigned int z;
} UVec3;

typedef struct {
	unsigned int x;
	unsigned int y;
	unsigned int z;
	unsigned int w;
} UVec4;

typedef struct {
	enum {
		INT,
		UINT,
		FLOAT,
		IVEC2,
		IVEC3,
		IVEC4,
		UVEC2,
		UVEC3,
		UVEC4,
		VEC2,
		VEC3,
		VEC4
	};
} AttributeType;

typedef int VertexHandle;
typedef int FaceHandle;
typedef int SphereHandle;
typedef void* ObjectHandle;
typedef void* AttributeHandle;

// Interface for objects
LIBRARY_API ObjectHandle polygon_create();

// Interface for polygons
LIBRARY_API void polygon_resize(ObjectHandle obj, uint64_t vertices, uint64_t edges,
								uint64_t faces);
LIBRARY_API AttributeHandle polygon_request_vertex_attribute(ObjectHandle obj, const char* name,
												  AttributeType type);
LIBRARY_API AttributeHandle polygon_request_face_attribute(ObjectHandle obj, const char* name,
												  AttributeType type);
LIBRARY_API void polygon_remove_vertex_attribute(ObjectHandle obj, ObjectHandle attr);
LIBRARY_API void polygon_remove_face_attribute(ObjectHandle obj, ObjectHandle attr);
LIBRARY_API AttributeHandle polygon_find_vertex_attribute(ObjectHandle obj,
														  const char* name,
														  AttributeType type);
LIBRARY_API AttributeHandle polygon_find_face_attribute(ObjectHandle obj,
														const char* name,
														AttributeType type);
LIBRARY_API int polygon_add_vertex(ObjectHandle obj, Vec3 point, Vec3 normal, Vec2 uv);
LIBRARY_API int polygon_add_triangle(ObjectHandle obj, UVec3 vertices);
LIBRARY_API int polygon_add_quad(ObjectHandle obj, UVec4 vertices);
LIBRARY_API int polygon_add_vertex_bulk(ObjectHandle obj, uint64_t count, FILE* points,
										FILE* normals, FILE* uvStream,
										uint64_t* pointsRead, uint64_t* normalsRead,
										uint64_t* uvsRead);
LIBRARY_API int polygon_add_vertex_bulk_aabb(ObjectHandle obj, uint64_t count, FILE* points,
										FILE* normals, FILE* uvStream,
										Vec3 min, Vec3 max, uint64_t* pointsRead,
										uint64_t* normalsRead, uint64_t* uvsRead);
LIBRARY_API void polygon_set_vertex_attribute(ObjectHandle obj, AttributeHandle attr,
											 VertexHandle vertex, AttributeType type,
											 void* value);
LIBRARY_API void polygon_set_face_attribute(ObjectHandle obj, AttributeHandle attr,
										   FaceHandle face, AttributeType type,
										   void* value);
LIBRARY_API int polygon_set_vertex_attribute_bulk(ObjectHandle obj, ObjectHandle attr,
												  VertexHandle startVertex,
												  uint64_t bytes, FILE* stream,
												  uint64_t* read);
LIBRARY_API int polygon_set_face_attribute_bulk(ObjectHandle obj, ObjectHandle attr,
												VertexHandle startVertex,
												uint64_t bytes, FILE* stream,
												uint64_t* read);
LIBRARY_API int64_t polygon_get_vertex_count(ObjectHandle obj);
LIBRARY_API int64_t polygon_get_edge_count(ObjectHandle obj);
LIBRARY_API int64_t polygon_get_face_count(ObjectHandle obj);
LIBRARY_API int64_t polygon_get_triangle_count(ObjectHandle obj);
LIBRARY_API int64_t polygon_get_quad_count(ObjectHandle obj);
LIBRARY_API void polygon_get_bounding_box(ObjectHandle obj, Vec3* min, Vec3* max);

// Interface for spheres
LIBRARY_API void spheres_resize(ObjectHandle obj, uint64_t count);
LIBRARY_API ObjectHandle spheres_request_attribute(ObjectHandle obj, const char* name,
											AttributeType type);
LIBRARY_API ObjectHandle spheres_remove_attribute(ObjectHandle obj, AttributeType attr);
LIBRARY_API ObjectHandle spheres_find_attribute(ObjectHandle obj, const char* name,
												AttributeType type);
LIBRARY_API SphereHandle spheres_add_sphere(ObjectHandle obj, Vec3 point, float radius);
LIBRARY_API SphereHandle spheres_add_sphere_bulk(ObjectHandle obj, uint64_t count,
												 FILE* stream, uint64_t* readSpheres);
LIBRARY_API SphereHandle spheres_add_sphere_bulk_aabb(ObjectHandle obj, uint64_t count,
													  FILE* stream, Vec3 min, Vec3 max,
													  uint64_t* readSpheres);
LIBRARY_API void spheres_set_attribute(ObjectHandle obj, AttributeHandle attr,
									   SphereHandle sphere, AttributeType type);
LIBRARY_API int spheres_set_attribute_bulk(ObjectHandle obj, AttributeHandle attr,
										   SphereHandle startSphere, uint64_t bytes,
										   FILE* stream, uint64_t* read);
LIBRARY_API int64_t spheres_get_sphere_count(ObjectHandle obj);
LIBRARY_API void spheres_get_bounding_box(ObjectHandle obj, Vec3* min, Vec3* max);

}