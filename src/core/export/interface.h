#pragma once

#include "api.h"
#include "texture_data.h"

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
	float x;
	float y;
	float z;
	float w;
} Vec4;

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

typedef struct {
	float v[12u];
} Mat3x4;

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
	MAT_LAMBERT,
	MAT_COUNT
} MaterialType;

typedef enum {
	CAM_PINHOLE,
	CAM_COUNT
} CameraType;

typedef enum {
	RENDERER_CPU_PT,
	RENDERER_GPU_PT,
	RENDERER_COUNT
} RendererType;

typedef enum {
	TARGET_RADIANCE,
	TARGET_POSITION,
	TARGET_ALBEDO,
	TARGET_NORMAL,
	TARGET_LIGHTNESS,
	TARGET_COUNT
} RenderTarget;

typedef enum {
	LIGHT_POINT,
	LIGHT_SPOT,
	LIGHT_DIRECTIONAL,
	LIGHT_ENVMAP,
	LIGHT_COUNT
} LightType;

typedef enum {
	SAMPLING_NEAREST,
	SAMPLING_LINEAR
} TextureSampling;

typedef enum {
	NDF_BECKMANN,
	NDF_GGX,
	NDF_PHONG
} NormalDistFunction;

typedef enum {
	MATERIAL_LAMBERT,
	MATERIAL_TORRANCE,
	MATERIAL_WALTER,
	MATERIAL_EMISSIVE,
	MATERIAL_ORENNAYAR,
	MATERIAL_BLEND,
	MATERIAL_FRESNEL,
	// MATERIAL_FRESNEL_CONDUCTOR	// Maybe reintroduce
} MaterialParamType;

typedef enum {
	MEDIUM_NONE,
	MEDIUM_DIELECTRIC,
	MEDIUM_CONDUCTOR
} OuterMediumType;

typedef enum {
	PROFILING_NONE,
	PROFILING_LOW,
	PROFILING_HIGH,
	PROFILING_ALL,
} ProfilingLevel;

typedef enum {
	NONE = 0,
	OBJ_EMISSIVE = 1
} ObjectFlags;

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

// Material types
typedef struct {
	Vec2 refractionIndex;
	Vec3 absorption;
} Medium;

typedef struct {
	TextureHdl albedo;
} LambertParams;
typedef struct {
	TextureHdl roughness;
	NormalDistFunction ndf;
	TextureHdl albedo;
} TorranceParams;
typedef struct {
	TextureHdl roughness;
	NormalDistFunction ndf;
	Vec3 absorption;
} WalterParams;
typedef struct {
	TextureHdl radiance;
	float scale;
} EmissiveParams;
typedef struct {
	TextureHdl albedo;
	float roughness;
} OrennayarParams;

// Forward declaration for recursive definition
struct MaterialParamsStruct;

typedef struct {
	typedef struct {
		float factor;
		struct MaterialParamsStruct* mat;
	} Layer;
	Layer a;
	Layer b;
} BlendParams;
typedef struct {
	Vec2 refractionIndex;
	struct MaterialParamsStruct* a;
	struct MaterialParamsStruct* b;
} FresnelParams;

typedef struct MaterialParamsStruct {
	Medium outerMedium;
	MaterialParamType innerType;
	union {
		LambertParams lambert;
		TorranceParams torrance;
		WalterParams walter;
		EmissiveParams emissive;
		OrennayarParams orennayar;
		BlendParams blend;
		FresnelParams fresnel;
	} inner;
} MaterialParams;

// Renderer parameter data
enum ParameterType {
	PARAM_INT,
	PARAM_FLOAT,
	PARAM_BOOL
};

// TODO: how to handle errors

// Polygon interface
CORE_API Boolean CDECL CDECL polygon_resize(ObjectHdl obj, size_t vertices, size_t edges,
											size_t tris, size_t quads);
CORE_API PolygonAttributeHandle CDECL polygon_request_vertex_attribute(ObjectHdl obj,
																	const char* name,
																	AttribDesc type);
CORE_API PolygonAttributeHandle CDECL polygon_request_face_attribute(ObjectHdl obj,
																  const char* name,
																  AttribDesc type);
CORE_API Boolean CDECL polygon_remove_vertex_attribute(ObjectHdl obj,
												 const PolygonAttributeHandle* hdl);
CORE_API Boolean CDECL polygon_remove_face_attribute(ObjectHdl obj,
											   const PolygonAttributeHandle* hdl);
CORE_API PolygonAttributeHandle CDECL polygon_find_vertex_attribute(ObjectHdl obj,
																 const char* name,
																 AttribDesc type);
CORE_API PolygonAttributeHandle CDECL polygon_find_face_attribute(ObjectHdl obj,
															   const char* name,
															   AttribDesc type);
CORE_API VertexHdl CDECL polygon_add_vertex(ObjectHdl obj, Vec3 point, Vec3 normal, Vec2 uv);
CORE_API FaceHdl CDECL polygon_add_triangle(ObjectHdl obj, UVec3 vertices);
CORE_API FaceHdl CDECL polygon_add_triangle_material(ObjectHdl obj, UVec3 vertices,
													 MatIdx idx);
CORE_API FaceHdl CDECL polygon_add_quad(ObjectHdl obj, UVec4 vertices);
CORE_API FaceHdl CDECL polygon_add_quad_material(ObjectHdl obj, UVec4 vertices,
												 MatIdx idx);
CORE_API VertexHdl CDECL polygon_add_vertex_bulk(ObjectHdl obj, size_t count, FILE* points,
										FILE* normals, FILE* uvs,
										size_t* pointsRead, size_t* normalsRead,
										size_t* uvsRead);
CORE_API VertexHdl CDECL polygon_add_vertex_bulk_no_normals(ObjectHdl obj, size_t count, FILE* points,
															FILE* uvs, size_t* pointsRead,
															size_t* uvsRead);
CORE_API VertexHdl CDECL polygon_add_vertex_bulk_aabb(ObjectHdl obj, size_t count, FILE* points,
										FILE* normals, FILE* uvs,
										Vec3 min, Vec3 max, size_t* pointsRead,
										size_t* normalsRead, size_t* uvsRead);
CORE_API VertexHdl CDECL polygon_add_vertex_bulk_aabb_no_normals(ObjectHdl obj, size_t count, FILE* points,
																 FILE* uvs, Vec3 min, Vec3 max,
																 size_t* pointsRead, size_t* uvsRead);
CORE_API Boolean CDECL polygon_set_vertex_attribute(ObjectHdl obj, const PolygonAttributeHandle* attr,
													VertexHdl vertex, const void* value);
CORE_API Boolean CDECL polygon_set_vertex_normal(ObjectHdl obj, VertexHdl vertex, Vec3 normal);
CORE_API Boolean CDECL polygon_set_vertex_uv(ObjectHdl obj, VertexHdl vertex, Vec2 uv);
CORE_API Boolean CDECL polygon_set_face_attribute(ObjectHdl obj, const PolygonAttributeHandle* attr,
											FaceHdl face, const void* value);
CORE_API Boolean CDECL polygon_set_material_idx(ObjectHdl obj, FaceHdl face, MatIdx idx);
CORE_API size_t CDECL polygon_set_vertex_attribute_bulk(ObjectHdl obj,
														const PolygonAttributeHandle* attr,
														VertexHdl startVertex, size_t count,
														FILE* stream);
CORE_API size_t CDECL polygon_set_face_attribute_bulk(ObjectHdl obj,
												   const PolygonAttributeHandle* attr,
												   FaceHdl startFace, size_t count,
												   FILE* stream);
CORE_API size_t CDECL polygon_set_material_idx_bulk(ObjectHdl obj, FaceHdl startFace, size_t count,
												 FILE* stream);
CORE_API size_t CDECL polygon_get_vertex_count(ObjectHdl obj);
CORE_API size_t CDECL polygon_get_edge_count(ObjectHdl obj);
CORE_API size_t CDECL polygon_get_face_count(ObjectHdl obj);
CORE_API size_t CDECL polygon_get_triangle_count(ObjectHdl obj);
CORE_API size_t CDECL polygon_get_quad_count(ObjectHdl obj);
CORE_API Boolean CDECL polygon_get_bounding_box(ObjectHdl obj, Vec3* min, Vec3* max);

// Spheres interface
CORE_API Boolean CDECL spheres_resize(ObjectHdl obj, size_t count);
CORE_API SphereAttributeHandle CDECL spheres_request_attribute(ObjectHdl obj,
															const char* name,
															AttribDesc type);
CORE_API Boolean CDECL spheres_remove_attribute(ObjectHdl obj, const SphereAttributeHandle* hdl);
CORE_API SphereAttributeHandle CDECL spheres_find_attribute(ObjectHdl obj,
														 const char* name,
														 AttribDesc type);
CORE_API SphereHdl CDECL spheres_add_sphere(ObjectHdl obj, Vec3 point, float radius);
CORE_API SphereHdl CDECL spheres_add_sphere_material(ObjectHdl obj, Vec3 point,
													 float radius, MatIdx idx);
CORE_API SphereHdl CDECL spheres_add_sphere_bulk(ObjectHdl obj, size_t count,
												 FILE* stream, size_t* readSpheres);
CORE_API SphereHdl CDECL spheres_add_sphere_bulk_aabb(ObjectHdl obj, size_t count,
													  FILE* stream, Vec3 min, Vec3 max,
													  size_t* readSpheres);
CORE_API Boolean CDECL spheres_set_attribute(ObjectHdl obj, const SphereAttributeHandle* attr,
									   SphereHdl sphere, const void* value);
CORE_API Boolean CDECL spheres_set_material_idx(ObjectHdl obj, SphereHdl sphere,
										  MatIdx idx);
CORE_API size_t CDECL spheres_set_attribute_bulk(ObjectHdl obj, const SphereAttributeHandle* attr,
											  SphereHdl startSphere, size_t count,
											  FILE* stream);
CORE_API size_t CDECL spheres_set_material_idx_bulk(ObjectHdl obj, SphereHdl startSphere,
												 size_t count, FILE* stream);
CORE_API size_t CDECL spheres_get_sphere_count(ObjectHdl obj);
CORE_API Boolean CDECL spheres_get_bounding_box(ObjectHdl obj, Vec3* min, Vec3* max);

// Instance interface
CORE_API Boolean CDECL instance_set_transformation_matrix(InstanceHdl inst, const Mat3x4* mat);
CORE_API Boolean CDECL instance_get_transformation_matrix(InstanceHdl inst, Mat3x4* mat);
CORE_API Boolean CDECL instance_get_bounding_box(InstanceHdl inst, Vec3* min, Vec3* max);

// World container interface
CORE_API void CDECL world_clear_all();
CORE_API ObjectHdl CDECL world_create_object(const char* name, ObjectFlags flags);
CORE_API ObjectHdl CDECL world_get_object(const char* name);
CORE_API InstanceHdl CDECL world_create_instance(ObjectHdl obj);
CORE_API ScenarioHdl CDECL world_create_scenario(const char* name);
CORE_API ScenarioHdl CDECL world_find_scenario(const char* name);
CORE_API MaterialHdl CDECL world_add_material(const char* name, const MaterialParams* mat);
// TODO: blended/fresnel materials
// TODO: glass/opaque materials
// TODO: add more cameras
CORE_API CameraHdl CDECL world_add_pinhole_camera(const char* name, Vec3 position,
											   Vec3 dir, Vec3 up, float near,
											   float far, float vFov);
CORE_API CameraHdl CDECL world_add_focus_camera(const char* name, Vec3 position, Vec3 dir,
												Vec3 up, float near, float far,
												float focalLength, float focusDistance,
												float lensRad, float chipHeight);
CORE_API LightHdl CDECL world_add_point_light(const char* name, Vec3 position,
										   Vec3 intensity);
CORE_API LightHdl CDECL world_add_spot_light(const char* name, Vec3 position,
										  Vec3 direction, Vec3 intensity,
										  float openingAngleRad,
										  float falloffStartRad);
CORE_API LightHdl CDECL world_add_directional_light(const char* name,
												 Vec3 direction,
												 Vec3 radiance);
CORE_API LightHdl CDECL world_add_envmap_light(const char* name, TextureHdl envmap);
CORE_API CameraHdl CDECL world_get_camera(const char* name);
CORE_API LightHdl CDECL world_get_light(const char* name, LightType type);
CORE_API SceneHdl CDECL world_load_scenario(ScenarioHdl scenario);
CORE_API SceneHdl CDECL world_get_current_scene();

// Scenario interface
CORE_API const char* CDECL scenario_get_name(ScenarioHdl scenario);
CORE_API LodLevel CDECL scenario_get_global_lod_level(ScenarioHdl scenario);
CORE_API Boolean CDECL scenario_set_global_lod_level(ScenarioHdl scenario, LodLevel level);
CORE_API Boolean CDECL scenario_get_resolution(ScenarioHdl scenario, uint32_t* width, uint32_t* height);
CORE_API Boolean CDECL scenario_set_resolution(ScenarioHdl scenario, uint32_t width, uint32_t height);
CORE_API CameraHdl CDECL scenario_get_camera(ScenarioHdl scenario);
CORE_API Boolean CDECL scenario_set_camera(ScenarioHdl scenario, CameraHdl cam);
CORE_API Boolean CDECL scenario_is_object_masked(ScenarioHdl scenario, ObjectHdl obj);
CORE_API Boolean CDECL scenario_mask_object(ScenarioHdl scenario, ObjectHdl obj);
CORE_API LodLevel CDECL scenario_get_object_lod(ScenarioHdl scenario, ObjectHdl obj);
CORE_API Boolean CDECL scenario_set_object_lod(ScenarioHdl scenario, ObjectHdl obj,
										 LodLevel level);
CORE_API IndexType CDECL scenario_get_light_count(ScenarioHdl scenario);
CORE_API const char* CDECL scenario_get_light_name(ScenarioHdl scenario, size_t index);
CORE_API Boolean CDECL scenario_add_light(ScenarioHdl scenario, const char* name);
CORE_API Boolean CDECL scenario_remove_light_by_index(ScenarioHdl scenario, size_t index);
CORE_API Boolean CDECL scenario_remove_light_by_named(ScenarioHdl scenario, const char* name);
CORE_API MatIdx CDECL scenario_declare_material_slot(ScenarioHdl scenario,
													 const char* name, size_t nameLength);
CORE_API MatIdx CDECL scenario_get_material_slot(ScenarioHdl scenario,
												 const char* name, size_t nameLength);
CORE_API MaterialHdl CDECL scenario_get_assigned_material(ScenarioHdl scenario,
														  MatIdx index);
CORE_API Boolean CDECL scenario_assign_material(ScenarioHdl scenario, MatIdx index,
										  MaterialHdl handle);

// Scene interface
CORE_API Boolean CDECL scene_get_bounding_box(SceneHdl scene, Vec3* min, Vec3* max);
CORE_API ConstCameraHdl CDECL scene_get_camera(SceneHdl scene);

// Light interface
CORE_API Boolean CDECL world_get_point_light_position(LightHdl hdl, Vec3* pos);
CORE_API Boolean CDECL world_get_point_light_intensity(LightHdl hdl, Vec3* intensity);
CORE_API Boolean CDECL world_set_point_light_position(LightHdl hdl, Vec3 pos);
CORE_API Boolean CDECL world_set_point_light_intensity(LightHdl hdl, Vec3 intensity);
CORE_API Boolean CDECL world_get_spot_light_position(LightHdl hdl, Vec3* pos);
CORE_API Boolean CDECL world_get_spot_light_intensity(LightHdl hdl, Vec3* intensity);
CORE_API Boolean CDECL world_get_spot_light_direction(LightHdl hdl, Vec3* direction);
CORE_API Boolean CDECL world_get_spot_light_angle(LightHdl hdl, float* angle);
CORE_API Boolean CDECL world_get_spot_light_falloff(LightHdl hdl, float* falloff);
CORE_API Boolean CDECL world_set_spot_light_position(LightHdl hdl, Vec3 pos);
CORE_API Boolean CDECL world_set_spot_light_intensity(LightHdl hdl, Vec3 intensity);
CORE_API Boolean CDECL world_set_spot_light_direction(LightHdl hdl, Vec3 direction);
CORE_API Boolean CDECL world_set_spot_light_angle(LightHdl hdl, float angle);
CORE_API Boolean CDECL world_set_spot_light_falloff(LightHdl hdl, float fallof);
CORE_API Boolean CDECL world_get_dir_light_direction(LightHdl hdl, Vec3* direction);
CORE_API Boolean CDECL world_get_dir_light_radiance(LightHdl hdl, Vec3* radiance);
CORE_API Boolean CDECL world_set_dir_light_direction(LightHdl hdl, Vec3 direction);
CORE_API Boolean CDECL world_set_dir_light_radiance(LightHdl hdl, Vec3 radiance);
CORE_API TextureHdl CDECL world_get_texture(const char* path);
CORE_API TextureHdl CDECL world_add_texture(const char* path, TextureSampling sampling);
CORE_API TextureHdl CDECL world_add_texture_value(const float* value, int num, TextureSampling sampling);
// TODO: interface for envmap light

// Interface for rendering
CORE_API Boolean CDECL render_enable_renderer(RendererType type);
CORE_API Boolean CDECL render_iterate();
CORE_API Boolean CDECL render_reset();
// TODO: what do we pass to the GUI?
CORE_API Boolean CDECL render_get_screenshot();
CORE_API Boolean CDECL render_save_screenshot(const char* filename);
CORE_API Boolean CDECL render_enable_render_target(RenderTarget target, Boolean variance);
CORE_API Boolean CDECL render_disable_render_target(RenderTarget target, Boolean variance);
CORE_API Boolean CDECL render_enable_variance_render_targets();
CORE_API Boolean CDECL render_enable_non_variance_render_targets();
CORE_API Boolean CDECL render_enable_all_render_targets();
CORE_API Boolean CDECL render_disable_variance_render_targets();
CORE_API Boolean CDECL render_disable_non_variance_render_targets();
CORE_API Boolean CDECL render_disable_all_render_targets();
CORE_API uint32_t CDECL renderer_get_num_parameters();
CORE_API const char* CDECL renderer_get_parameter_desc(uint32_t idx, ParameterType* type);
CORE_API Boolean CDECL renderer_set_parameter_int(const char* name, int32_t value);
CORE_API Boolean CDECL renderer_get_parameter_int(const char* name, int32_t* value);
CORE_API Boolean CDECL renderer_set_parameter_float(const char* name, float value);
CORE_API Boolean CDECL renderer_get_parameter_float(const char* name, float* value);
CORE_API Boolean CDECL renderer_set_parameter_bool(const char* name, Boolean value);
CORE_API Boolean CDECL renderer_get_parameter_bool(const char* name, Boolean* value);

// Interface for profiling
CORE_API void CDECL profiling_enable();
CORE_API void CDECL profiling_disable();
CORE_API Boolean CDECL profiling_set_level(ProfilingLevel level);
CORE_API Boolean CDECL profiling_save_current_state(const char* path);
CORE_API Boolean CDECL profiling_save_snapshots(const char* path);
CORE_API Boolean CDECL profiling_save_total_and_snapshots(const char* path);
CORE_API const char* CDECL profiling_get_current_state();
CORE_API const char* CDECL profiling_get_snapshots();
CORE_API const char* CDECL profiling_get_total_and_snapshots();
CORE_API void CDECL profiling_reset();
CORE_API size_t CDECL profiling_get_total_cpu_memory();
CORE_API size_t CDECL profiling_get_free_cpu_memory();
CORE_API size_t CDECL profiling_get_used_cpu_memory();
CORE_API size_t CDECL profiling_get_total_gpu_memory();
CORE_API size_t CDECL profiling_get_free_gpu_memory();
CORE_API size_t CDECL profiling_get_used_gpu_memory();

// Interface for initialization and destruction
CORE_API Boolean CDECL mufflon_initialize(void(*logCallback)(const char*, int));
CORE_API void CDECL mufflon_destroy();

// TODO
CORE_API Boolean CDECL display_screenshot();
CORE_API Boolean CDECL resize(int width, int height, int offsetX, int offsetY);
CORE_API void CDECL execute_command(const char* command);
CORE_API const char* CDECL get_error(int& length);


}
