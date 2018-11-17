#pragma once

#include "api.hpp"

extern "C" {

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <limits.h>

#define INVALID_INDEX int32_t{-1}
#define INVALID_SIZE ULLONG_MAX
#define INVALID_MATERIAL USHRT_MAX
// Typedef for boolean value (since the standard doesn't specify
// its size
typedef uint32_t Boolean;

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
	uint32_t x;
	uint32_t y;
	uint32_t z;
} UVec3;

typedef struct {
	uint32_t x;
	uint32_t y;
	uint32_t z;
	uint32_t w;
} UVec4;

typedef struct {
	int32_t x;
	int32_t y;
} IVec2;

typedef enum {
	ATTR_CHAR,
	ATTR_UCHAR,
	ATTR_SHORT,
	ATTR_USHORT,
	ATTR_INT,
	ATTR_UINT,
	ATTR_LONG,
	ATTR_ULONG,
	ATTR_FLOAT,
	ATTR_DOUBLE,
	ATTR_COUNT
} AttributeType;

typedef enum {
	MAT_LAMBERT
} MaterialType;

typedef enum {
	CAM_PINHOLE
} CameraType;

typedef enum {
	RENDERER_CPU_PT,
	RENDERER_GPU_PT,
} RendererType;

typedef enum {
	TARGET_RADIANCE,
	TARGET_POSITION,
	TARGET_ALBEDO,
	TARGET_NORMAL,
	TARGET_LIGHTNESS
} RenderTarget;

typedef enum {
	LIGHT_POINT,
	LIGHT_SPOT,
	LIGHT_DIRECTIONAL,
	LIGHT_ENVMAP
} LightType;

typedef struct {
	AttributeType type;
	uint32_t rows;
} AttribDesc;

typedef struct {
	int32_t openMeshIndex;
	int32_t customIndex;
	AttribDesc type;
	Boolean face;
} PolygonAttributeHandle;

typedef struct {
	int32_t index;
	AttribDesc type;
} SphereAttributeHandle;

// Typedefs for return values
// Not to be accessed directly!
typedef int32_t IndexType;
typedef IndexType VertexHdl;
typedef IndexType FaceHdl;
typedef IndexType SphereHdl;
typedef uint64_t LodLevel;
typedef uint16_t MatIdx;
typedef void* ObjectHdl;
typedef void* InstanceHdl;
typedef void* ScenarioHdl;
typedef void* SceneHdl;
typedef void* MaterialHdl;
typedef void* CameraHdl;
typedef void* LightHdl;
typedef void* TextureHdl;
typedef const void* ConstCameraHdl;

// TODO: how to handle errors

// Polygon interface
LIBRARY_API Boolean polygon_resize(ObjectHdl obj, size_t vertices, size_t edges,
								size_t faces);
LIBRARY_API PolygonAttributeHandle polygon_request_vertex_attribute(ObjectHdl obj,
																	const char* name,
																	AttribDesc type);
LIBRARY_API PolygonAttributeHandle polygon_request_face_attribute(ObjectHdl obj,
																  const char* name,
																  AttribDesc type);
LIBRARY_API Boolean polygon_remove_vertex_attribute(ObjectHdl obj,
												 const PolygonAttributeHandle* hdl);
LIBRARY_API Boolean polygon_remove_face_attribute(ObjectHdl obj,
											   const PolygonAttributeHandle* hdl);
LIBRARY_API PolygonAttributeHandle polygon_find_vertex_attribute(ObjectHdl obj,
																 const char* name,
																 AttribDesc type);
LIBRARY_API PolygonAttributeHandle polygon_find_face_attribute(ObjectHdl obj,
															   const char* name,
															   AttribDesc type);
LIBRARY_API VertexHdl polygon_add_vertex(ObjectHdl obj, Vec3 point, Vec3 normal, Vec2 uv);
LIBRARY_API FaceHdl polygon_add_triangle(ObjectHdl obj, UVec3 vertices);
LIBRARY_API FaceHdl polygon_add_triangle_material(ObjectHdl obj, UVec3 vertices,
													 MatIdx idx);
LIBRARY_API FaceHdl polygon_add_quad(ObjectHdl obj, UVec4 vertices);
LIBRARY_API FaceHdl polygon_add_quad_material(ObjectHdl obj, UVec4 vertices,
												 MatIdx idx);
LIBRARY_API VertexHdl polygon_add_vertex_bulk(ObjectHdl obj, size_t count, FILE* points,
										FILE* normals, FILE* uvStream,
										size_t* pointsRead, size_t* normalsRead,
										size_t* uvsRead);
LIBRARY_API VertexHdl polygon_add_vertex_bulk_aabb(ObjectHdl obj, size_t count, FILE* points,
										FILE* normals, FILE* uvStream,
										Vec3 min, Vec3 max, size_t* pointsRead,
										size_t* normalsRead, size_t* uvsRead);
LIBRARY_API Boolean polygon_set_vertex_attribute(ObjectHdl obj, const PolygonAttributeHandle* attr,
											  VertexHdl vertex, void* value);
LIBRARY_API Boolean polygon_set_face_attribute(ObjectHdl obj, const PolygonAttributeHandle* attr,
											FaceHdl face, void* value);
LIBRARY_API Boolean polygon_set_material_idx(ObjectHdl obj, FaceHdl face, MatIdx idx);
LIBRARY_API size_t polygon_set_vertex_attribute_bulk(ObjectHdl obj,
													 const PolygonAttributeHandle* attr,
													 VertexHdl startVertex, AttribDesc type,
													 size_t count, FILE* stream);
LIBRARY_API size_t polygon_set_face_attribute_bulk(ObjectHdl obj,
												   const PolygonAttributeHandle* attr,
												   FaceHdl startFace, size_t count,
												   FILE* stream);
LIBRARY_API size_t polygon_set_material_idx_bulk(ObjectHdl obj, FaceHdl startFace, size_t count,
												 FILE* stream);
LIBRARY_API size_t polygon_get_vertex_count(ObjectHdl obj);
LIBRARY_API size_t polygon_get_edge_count(ObjectHdl obj);
LIBRARY_API size_t polygon_get_face_count(ObjectHdl obj);
LIBRARY_API size_t polygon_get_triangle_count(ObjectHdl obj);
LIBRARY_API size_t polygon_get_quad_count(ObjectHdl obj);
LIBRARY_API Boolean polygon_get_bounding_box(ObjectHdl obj, Vec3* min, Vec3* max);

// Spheres interface
LIBRARY_API Boolean spheres_resize(ObjectHdl obj, size_t count);
LIBRARY_API SphereAttributeHandle spheres_request_attribute(ObjectHdl obj,
															const char* name,
															AttribDesc type);
LIBRARY_API Boolean spheres_remove_attribute(ObjectHdl obj, const SphereAttributeHandle* hdl);
LIBRARY_API SphereAttributeHandle spheres_find_attribute(ObjectHdl obj,
														 const char* name,
														 AttribDesc type);
LIBRARY_API SphereHdl spheres_add_sphere(ObjectHdl obj, Vec3 point, float radius);
LIBRARY_API SphereHdl spheres_add_sphere_material(ObjectHdl obj, Vec3 point,
													 float radius, MatIdx idx);
LIBRARY_API SphereHdl spheres_add_sphere_bulk(ObjectHdl obj, size_t count,
												 FILE* stream, size_t* readSpheres);
LIBRARY_API SphereHdl spheres_add_sphere_bulk_aabb(ObjectHdl obj, size_t count,
													  FILE* stream, Vec3 min, Vec3 max,
													  size_t* readSpheres);
LIBRARY_API Boolean spheres_set_attribute(ObjectHdl obj, const SphereAttributeHandle* attr,
									   SphereHdl sphere, void* value);
LIBRARY_API Boolean spheres_set_material_idx(ObjectHdl obj, SphereHdl sphere,
										  MatIdx idx);
LIBRARY_API size_t spheres_set_attribute_bulk(ObjectHdl obj, const SphereAttributeHandle* attr,
											  SphereHdl startSphere, size_t count,
											  FILE* stream);
LIBRARY_API size_t spheres_set_material_idx_bulk(ObjectHdl obj, SphereHdl startSphere,
												 size_t count, FILE* stream);
LIBRARY_API size_t spheres_get_sphere_count(ObjectHdl obj);
LIBRARY_API Boolean spheres_get_bounding_box(ObjectHdl obj, Vec3* min, Vec3* max);

// World container interface
LIBRARY_API ObjectHdl world_create_object();
LIBRARY_API InstanceHdl world_create_instance(ObjectHdl obj);
LIBRARY_API ScenarioHdl world_create_scenario(const char* name);
LIBRARY_API ScenarioHdl world_find_scenario(const char* name);
// TODO: add more materials (and material parameters)
LIBRARY_API MaterialHdl world_add_lambert_material(const char* name);
// TODO: add more cameras
LIBRARY_API CameraHdl world_add_pinhole_camera(const char* name, Vec3 position,
											   Vec3 dir, Vec3 up, float near,
											   float far, float vFov);
LIBRARY_API LightHdl world_add_point_light(const char* name, Vec3 position,
										   Vec3 intensity);
LIBRARY_API LightHdl world_add_spot_light(const char* name, Vec3 position,
										  Vec3 direction, Vec3 intensity,
										  float openingAngleRad,
										  float falloffStartRad);
LIBRARY_API LightHdl world_add_directional_light(const char* name,
												 Vec3 direction,
												 Vec3 radiance);
LIBRARY_API LightHdl world_add_envmap_light(const char* name, TextureHdl envmap);
LIBRARY_API CameraHdl world_get_camera(const char* name);
LIBRARY_API LightHdl world_get_light(const char* name, LightType type);
LIBRARY_API SceneHdl world_load_scenario(ScenarioHdl scenario);
LIBRARY_API SceneHdl world_get_current_scene();

// Scenario interface
LIBRARY_API const char* scenario_get_name(ScenarioHdl scenario);
LIBRARY_API LodLevel scenario_get_global_lod_level(ScenarioHdl scenario);
LIBRARY_API Boolean scenario_set_global_lod_level(ScenarioHdl scenario, LodLevel level);
LIBRARY_API IVec2 scenario_get_resolution(ScenarioHdl scenario);
LIBRARY_API Boolean scenario_set_resolution(ScenarioHdl scenario, IVec2 res);
LIBRARY_API CameraHdl scenario_get_camera(ScenarioHdl scenario);
LIBRARY_API Boolean scenario_set_camera(ScenarioHdl scenario, CameraHdl cam);
LIBRARY_API Boolean scenario_is_object_masked(ScenarioHdl scenario, ObjectHdl obj);
LIBRARY_API Boolean scenario_mask_object(ScenarioHdl scenario, ObjectHdl obj);
LIBRARY_API LodLevel scenario_get_object_lod(ScenarioHdl scenario, ObjectHdl obj);
LIBRARY_API Boolean scenario_set_object_lod(ScenarioHdl scenario, ObjectHdl obj,
										 LodLevel level);
LIBRARY_API IndexType scenario_get_light_count(ScenarioHdl scenario);
LIBRARY_API const char* scenario_get_light_name(ScenarioHdl scenario, size_t index);
LIBRARY_API Boolean scenario_add_light(ScenarioHdl scenario, const char* name);
LIBRARY_API Boolean scenario_remove_light_by_index(ScenarioHdl scenario, size_t index);
LIBRARY_API Boolean scenario_remove_light_by_named(ScenarioHdl scenario, const char* name);
LIBRARY_API MatIdx scenario_declare_material_slot(ScenarioHdl scenario,
														 const char* name);
LIBRARY_API MatIdx scenario_get_material_slot(ScenarioHdl scenario,
													 const char* name);
LIBRARY_API MaterialHdl scenario_get_assigned_material(ScenarioHdl scenario,
														  MatIdx index);
LIBRARY_API Boolean scenario_assign_material(ScenarioHdl scenario, MatIdx index,
										  MaterialHdl handle);

// Scene interface
LIBRARY_API Boolean scene_get_bounding_box(SceneHdl scene, Vec3* min, Vec3* max);
LIBRARY_API ConstCameraHdl scene_get_camera(SceneHdl scene);

// Light interface
LIBRARY_API Boolean world_get_point_light_position(LightHdl hdl, Vec3* pos);
LIBRARY_API Boolean world_get_point_light_intensity(LightHdl hdl, Vec3* intensity);
LIBRARY_API Boolean world_set_point_light_position(LightHdl hdl, Vec3 pos);
LIBRARY_API Boolean world_set_point_light_intensity(LightHdl hdl, Vec3 intensity);
LIBRARY_API Boolean world_get_spot_light_position(LightHdl hdl, Vec3* pos);
LIBRARY_API Boolean world_get_spot_light_intensity(LightHdl hdl, Vec3* intensity);
LIBRARY_API Boolean world_get_spot_light_direction(LightHdl hdl, Vec3* direction);
LIBRARY_API Boolean world_get_spot_light_angle(LightHdl hdl, float* angle);
LIBRARY_API Boolean world_get_spot_light_falloff(LightHdl hdl, float* falloff);
LIBRARY_API Boolean world_set_spot_light_position(LightHdl hdl, Vec3 pos);
LIBRARY_API Boolean world_set_spot_light_intensity(LightHdl hdl, Vec3 intensity);
LIBRARY_API Boolean world_set_spot_light_direction(LightHdl hdl, Vec3 direction);
LIBRARY_API Boolean world_set_spot_light_angle(LightHdl hdl, float angle);
LIBRARY_API Boolean world_set_spot_light_falloff(LightHdl hdl, float fallof);
LIBRARY_API Boolean world_get_dir_light_direction(LightHdl hdl, Vec3* direction);
LIBRARY_API Boolean world_get_dir_light_radiance(LightHdl hdl, Vec3* radiance);
LIBRARY_API Boolean world_set_dir_light_direction(LightHdl hdl, Vec3 direction);
LIBRARY_API Boolean world_set_dir_light_radiance(LightHdl hdl, Vec3 radiance);
// TODO: interface for envmap light

// Interface for rendering
LIBRARY_API Boolean render_enable_renderer(RendererType type);
LIBRARY_API Boolean render_iterate();
// TODO: what do we pass to the GUI?
LIBRARY_API Boolean render_get_screenshot();
LIBRARY_API Boolean render_save_screenshot(const char* filename);
LIBRARY_API Boolean render_enable_render_target(RenderTarget target, Boolean variance);
LIBRARY_API Boolean render_disable_render_target(RenderTarget target, Boolean variance);
LIBRARY_API Boolean render_enable_variance_render_targets();
LIBRARY_API Boolean render_enable_non_variance_render_targets();
LIBRARY_API Boolean render_enable_all_render_targets();
LIBRARY_API Boolean render_disable_variance_render_targets();
LIBRARY_API Boolean render_disable_non_variance_render_targets();
LIBRARY_API Boolean render_disable_all_render_targets();

// TODO
LIBRARY_API Boolean initialize(void(*logCallback)(const char*, int));
LIBRARY_API Boolean iterate();
LIBRARY_API Boolean resize(int width, int height, int offsetX, int offsetY);
LIBRARY_API void execute_command(const char* command);
LIBRARY_API const char* get_error(int& length);

}