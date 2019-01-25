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

typedef struct {
	Vec3 min;
	Vec3 max;
} AABB;

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
	CAM_FOCUS,
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
	NDF_COSINE
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
	MATERIAL_NUM,
} MaterialParamType;

typedef enum {
	MEDIUM_NONE,
	MEDIUM_DIELECTRIC,
	MEDIUM_CONDUCTOR
} OuterMediumType;

typedef enum {
	PROFILING_ALL,
	PROFILING_HIGH,
	PROFILING_LOW,
	PROFILING_OFF
} ProfilingLevel;

typedef enum {
	LOG_PEDANTIC,
	LOG_INFO,
	LOG_WARNING,
	LOG_ERROR,
	LOG_FATAL_ERROR
} LogLevel;

typedef enum {
	NONE = 0,
	OBJ_EMISSIVE = 1
} ObjectFlags;

typedef struct {
	AttributeType type;
	uint32_t rows;
} AttribDesc;

typedef struct {
	int32_t index;
	AttribDesc type;
	Boolean face;
} PolygonAttributeHdl;

typedef struct {
	int32_t index;
	AttribDesc type;
} SphereAttributeHdl;

typedef struct {
	uint32_t type: 3;
	uint32_t index: 29;
} LightHdl;

// Typedefs for return values
// Not to be accessed directly!
typedef int32_t IndexType;
typedef IndexType VertexHdl;
typedef IndexType FaceHdl;
typedef IndexType SphereHdl;
typedef uint32_t LodLevel;
typedef uint16_t MatIdx;
typedef void* ObjectHdl;
typedef void* ConstObjectHdl;
typedef void* LodHdl;
typedef void* InstanceHdl;
typedef void* ScenarioHdl;
typedef const void* ConstScenarioHdl;
typedef void* SceneHdl;
typedef void* MaterialHdl;
typedef void* CameraHdl;
typedef void* TextureHdl;
typedef const void* ConstCameraHdl;
typedef const LightHdl ConstLightHdl;

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
	Vec3 scale;
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

// Abstraction for bulk load
struct BulkLoader {
	enum BulkType {
		BULK_FILE,
		BULK_ARRAY
	} type;
	union {
		FILE* file;
		const char* bytes;
	} descriptor;
};

// TODO: how to handle errors

// Polygon interface
CORE_API Boolean CDECL CDECL polygon_reserve(LodHdl lod, size_t vertices, size_t edges,
											size_t tris, size_t quads);
CORE_API PolygonAttributeHdl CDECL polygon_request_vertex_attribute(LodHdl lod,
																	const char* name,
																	AttribDesc type);
CORE_API PolygonAttributeHdl CDECL polygon_request_face_attribute(LodHdl lod,
																  const char* name,
																  AttribDesc type);
CORE_API VertexHdl CDECL polygon_add_vertex(LodHdl lod, Vec3 point, Vec3 normal, Vec2 uv);
CORE_API FaceHdl CDECL polygon_add_triangle(LodHdl lod, UVec3 vertices);
CORE_API FaceHdl CDECL polygon_add_triangle_material(LodHdl lod, UVec3 vertices,
													 MatIdx idx);
CORE_API FaceHdl CDECL polygon_add_quad(LodHdl lod, UVec4 vertices);
CORE_API FaceHdl CDECL polygon_add_quad_material(LodHdl lod, UVec4 vertices,
												 MatIdx idx);
CORE_API VertexHdl CDECL polygon_add_vertex_bulk(LodHdl lod, size_t count, const BulkLoader* points,
												 const BulkLoader* normals, const BulkLoader* uvs,
												 const AABB* aabb, size_t* pointsRead, size_t* normalsRead,
												 size_t* uvsRead);
CORE_API Boolean CDECL polygon_set_vertex_attribute(LodHdl lod, const PolygonAttributeHdl* attr,
													VertexHdl vertex, const void* value);
CORE_API Boolean CDECL polygon_set_vertex_normal(LodHdl lod, VertexHdl vertex, Vec3 normal);
CORE_API Boolean CDECL polygon_set_vertex_uv(LodHdl lod, VertexHdl vertex, Vec2 uv);
CORE_API Boolean CDECL polygon_set_face_attribute(LodHdl lod, const PolygonAttributeHdl* attr,
											FaceHdl face, const void* value);
CORE_API Boolean CDECL polygon_set_material_idx(LodHdl lod, FaceHdl face, MatIdx idx);
CORE_API size_t CDECL polygon_set_vertex_attribute_bulk(LodHdl lod,
														const PolygonAttributeHdl* attr,
														VertexHdl startVertex, size_t count,
														const BulkLoader* stream);
CORE_API size_t CDECL polygon_set_face_attribute_bulk(LodHdl lod,
												   const PolygonAttributeHdl* attr,
												   FaceHdl startFace, size_t count,
													  const BulkLoader* stream);
CORE_API size_t CDECL polygon_set_material_idx_bulk(LodHdl lod, FaceHdl startFace, size_t count,
													const BulkLoader* stream);
CORE_API size_t CDECL polygon_get_vertex_count(LodHdl lod);
CORE_API size_t CDECL polygon_get_edge_count(LodHdl lod);
CORE_API size_t CDECL polygon_get_face_count(LodHdl lod);
CORE_API size_t CDECL polygon_get_triangle_count(LodHdl lod);
CORE_API size_t CDECL polygon_get_quad_count(LodHdl lod);
CORE_API Boolean CDECL polygon_get_bounding_box(LodHdl lod, Vec3* min, Vec3* max);

// Spheres interface
CORE_API Boolean CDECL spheres_reserve(LodHdl lod, size_t count);
CORE_API SphereAttributeHdl CDECL spheres_request_attribute(LodHdl lod,
															const char* name,
															AttribDesc type);
CORE_API SphereHdl CDECL spheres_add_sphere(LodHdl lod, Vec3 point, float radius);
CORE_API SphereHdl CDECL spheres_add_sphere_material(LodHdl lod, Vec3 point,
													 float radius, MatIdx idx);
CORE_API SphereHdl CDECL spheres_add_sphere_bulk(LodHdl lod, size_t count,
												 const BulkLoader* stream, const AABB* aabbb,
												 size_t* readSpheres);
CORE_API Boolean CDECL spheres_set_attribute(LodHdl lod, const SphereAttributeHdl* attr,
									   SphereHdl sphere, const void* value);
CORE_API Boolean CDECL spheres_set_material_idx(LodHdl lod, SphereHdl sphere,
										  MatIdx idx);
CORE_API size_t CDECL spheres_set_attribute_bulk(LodHdl lod, const SphereAttributeHdl* attr,
											  SphereHdl startSphere, size_t count,
											  const BulkLoader* stream);
CORE_API size_t CDECL spheres_set_material_idx_bulk(LodHdl lod, SphereHdl startSphere,
												 size_t count, const BulkLoader* stream);
CORE_API size_t CDECL spheres_get_sphere_count(LodHdl lod);
CORE_API Boolean CDECL spheres_get_bounding_box(LodHdl lod, Vec3* min, Vec3* max);

// Object interface
CORE_API Boolean CDECL object_has_lod(ObjectHdl hdl, LodLevel level);
CORE_API LodHdl CDECL object_add_lod(ObjectHdl hdl, LodLevel level);
CORE_API Boolean CDECL object_set_animation_frame(ObjectHdl hdl, uint32_t animFrame);
CORE_API Boolean CDECL object_get_animation_frame(ObjectHdl hdl, uint32_t* animFrame);
CORE_API Boolean CDECL object_get_id(ObjectHdl hdl, uint32_t* id);

// Instance interface
CORE_API Boolean CDECL instance_set_transformation_matrix(InstanceHdl inst, const Mat3x4* mat);
CORE_API Boolean CDECL instance_get_transformation_matrix(InstanceHdl inst, Mat3x4* mat);
CORE_API Boolean CDECL instance_get_bounding_box(InstanceHdl inst, Vec3* min, Vec3* max, LodLevel lod);

// World container interface
CORE_API void CDECL world_clear_all();
CORE_API ObjectHdl CDECL world_create_object(const char* name, ObjectFlags flags);
CORE_API ObjectHdl CDECL world_get_object(const char* name);
CORE_API InstanceHdl CDECL world_get_instance(const char* name);
CORE_API const char* CDECL world_get_object_name(ObjectHdl obj);
CORE_API InstanceHdl CDECL world_create_instance(const char* name, ObjectHdl obj);
CORE_API ScenarioHdl CDECL world_create_scenario(const char* name);
CORE_API ScenarioHdl CDECL world_find_scenario(const char* name);
CORE_API uint32_t CDECL world_get_scenario_count();
CORE_API ScenarioHdl CDECL world_get_scenario_by_index(uint32_t index);
CORE_API ConstScenarioHdl CDECL world_get_current_scenario();
CORE_API MaterialHdl CDECL world_add_material(const char* name, const MaterialParams* mat);
CORE_API IndexType CDECL world_get_material_count();
CORE_API MaterialHdl CDECL world_get_material(IndexType index);
CORE_API size_t CDECL world_get_material_size(MaterialHdl material);
// buffer must have at least world_get_material_size() bytes. After a successful get
// buffer contains a MaterialParamsStruct and all referenced sub-layers.
CORE_API Boolean CDECL world_get_material_data(MaterialHdl material, MaterialParams* buffer);
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
CORE_API Boolean CDECL world_remove_camera(CameraHdl hdl);
CORE_API LightHdl CDECL world_add_light(const char* name, LightType type);
CORE_API Boolean CDECL world_set_light_name(LightHdl hdl, const char* newName);
CORE_API Boolean CDECL world_remove_light(LightHdl hdl);
CORE_API Boolean CDECL world_find_light(const char* name, LightHdl* hdl);
CORE_API size_t CDECL world_get_camera_count();
CORE_API CameraHdl CDECL world_get_camera(const char* name);
CORE_API CameraHdl CDECL world_get_camera_by_index(size_t);
CORE_API size_t CDECL world_get_point_light_count();
CORE_API size_t CDECL world_get_spot_light_count();
CORE_API size_t CDECL world_get_dir_light_count();
CORE_API size_t CDECL world_get_env_light_count();
CORE_API LightHdl CDECL world_get_light_handle(size_t index, LightType type);
CORE_API const char* CDECL world_get_light_name(LightHdl hdl);
CORE_API SceneHdl CDECL world_load_scenario(ScenarioHdl scenario);
CORE_API SceneHdl CDECL world_get_current_scene();
CORE_API Boolean CDECL world_is_sane(const char** msg);

// Scenario interface
CORE_API const char* CDECL scenario_get_name(ScenarioHdl scenario);
CORE_API LodLevel CDECL scenario_get_global_lod_level(ScenarioHdl scenario);
CORE_API Boolean CDECL scenario_set_global_lod_level(ScenarioHdl scenario, LodLevel level);
CORE_API Boolean CDECL scenario_get_resolution(ScenarioHdl scenario, uint32_t* width, uint32_t* height);
CORE_API Boolean CDECL scenario_set_resolution(ScenarioHdl scenario, uint32_t width, uint32_t height);
CORE_API CameraHdl CDECL scenario_get_camera(ScenarioHdl scenario);
CORE_API Boolean CDECL scenario_set_camera(ScenarioHdl scenario, CameraHdl cam);
CORE_API Boolean CDECL scenario_is_object_masked(ScenarioHdl scenario, ObjectHdl obj);
CORE_API Boolean CDECL scenario_mask_object(ScenarioHdl scenario, ObjectHdl inst);
CORE_API Boolean CDECL scenario_mask_instance(ScenarioHdl scenario, InstanceHdl obj);
CORE_API LodLevel CDECL scenario_get_object_lod(ScenarioHdl scenario, ObjectHdl obj);
CORE_API Boolean CDECL scenario_set_object_lod(ScenarioHdl scenario, ObjectHdl obj,
										 LodLevel level);
CORE_API Boolean CDECL scenario_set_instance_lod(ScenarioHdl scenario, InstanceHdl inst,
										 LodLevel level);
CORE_API IndexType CDECL scenario_get_point_light_count(ScenarioHdl scenario);
CORE_API IndexType CDECL scenario_get_spot_light_count(ScenarioHdl scenario);
CORE_API IndexType CDECL scenario_get_dir_light_count(ScenarioHdl scenario);
CORE_API Boolean CDECL scenario_has_envmap_light(ScenarioHdl scenario);
CORE_API LightHdl CDECL scenario_get_light_handle(ScenarioHdl scenario, IndexType index, LightType type);
CORE_API Boolean CDECL scenario_add_light(ScenarioHdl scenario, LightHdl hdl);
CORE_API Boolean CDECL scenario_remove_light(ScenarioHdl scenario, LightHdl hdl);
CORE_API MatIdx CDECL scenario_declare_material_slot(ScenarioHdl scenario,
													 const char* name, size_t nameLength);
CORE_API MatIdx CDECL scenario_get_material_slot(ScenarioHdl scenario,
												 const char* name, size_t nameLength);
CORE_API MaterialHdl CDECL scenario_get_assigned_material(ScenarioHdl scenario,
														  MatIdx index);
CORE_API Boolean CDECL scenario_assign_material(ScenarioHdl scenario, MatIdx index,
												MaterialHdl handle);
CORE_API Boolean CDECL scenario_is_sane(ConstScenarioHdl, const char** msg);

// Scene interface
CORE_API Boolean CDECL scene_get_bounding_box(SceneHdl scene, Vec3* min, Vec3* max);
CORE_API ConstCameraHdl CDECL scene_get_camera(SceneHdl scene);
CORE_API Boolean CDECL scene_move_active_camera(float x, float y, float z);
CORE_API Boolean CDECL scene_rotate_active_camera(float x, float y, float z);
CORE_API Boolean CDECL scene_is_sane();

// Light interface
CORE_API Boolean CDECL world_get_point_light_position(ConstLightHdl hdl, Vec3* pos);
CORE_API Boolean CDECL world_get_point_light_intensity(ConstLightHdl hdl, Vec3* intensity);
CORE_API Boolean CDECL world_set_point_light_position(LightHdl hdl, Vec3 pos);
CORE_API Boolean CDECL world_set_point_light_intensity(LightHdl hdl, Vec3 intensity);
CORE_API Boolean CDECL world_get_spot_light_position(ConstLightHdl hdl, Vec3* pos);
CORE_API Boolean CDECL world_get_spot_light_intensity(ConstLightHdl hdl, Vec3* intensity);
CORE_API Boolean CDECL world_get_spot_light_direction(ConstLightHdl hdl, Vec3* direction);
CORE_API Boolean CDECL world_get_spot_light_angle(ConstLightHdl hdl, float* angle);
CORE_API Boolean CDECL world_get_spot_light_falloff(ConstLightHdl hdl, float* falloff);
CORE_API Boolean CDECL world_set_spot_light_position(LightHdl hdl, Vec3 pos);
CORE_API Boolean CDECL world_set_spot_light_intensity(LightHdl hdl, Vec3 intensity);
CORE_API Boolean CDECL world_set_spot_light_direction(LightHdl hdl, Vec3 direction);
CORE_API Boolean CDECL world_set_spot_light_angle(LightHdl hdl, float angle);
CORE_API Boolean CDECL world_set_spot_light_falloff(LightHdl hdl, float fallof);
CORE_API Boolean CDECL world_get_dir_light_direction(ConstLightHdl hdl, Vec3* direction);
CORE_API Boolean CDECL world_get_dir_light_irradiance(ConstLightHdl hdl, Vec3* irradiance);
CORE_API Boolean CDECL world_set_dir_light_direction(LightHdl hdl, Vec3 direction);
CORE_API Boolean CDECL world_set_dir_light_irradiance(LightHdl hdl, Vec3 irradiance);
CORE_API const char* CDECL world_get_env_light_map(ConstLightHdl hdl);
CORE_API Boolean CDECL world_get_env_light_scale(LightHdl hdl, Vec3* color);
CORE_API Boolean CDECL world_set_env_light_map(LightHdl hdl, TextureHdl tex);
CORE_API Boolean CDECL world_set_env_light_scale(LightHdl hdl, Vec3 color);
CORE_API TextureHdl CDECL world_get_texture(const char* path);
CORE_API TextureHdl CDECL world_add_texture(const char* path, TextureSampling sampling);
CORE_API TextureHdl CDECL world_add_texture_value(const float* value, int num, TextureSampling sampling);

// Camera interface
CORE_API CameraType CDECL world_get_camera_type(ConstCameraHdl cam);
CORE_API const char* CDECL world_get_camera_name(ConstCameraHdl cam);
CORE_API Boolean CDECL world_get_camera_position(ConstCameraHdl cam, Vec3* pos);
CORE_API Boolean CDECL world_get_camera_direction(ConstCameraHdl cam, Vec3* dir);
CORE_API Boolean CDECL world_get_camera_up(ConstCameraHdl cam, Vec3* up);
CORE_API Boolean CDECL world_get_camera_near(ConstCameraHdl cam, float* near);
CORE_API Boolean CDECL world_get_camera_far(ConstCameraHdl cam, float* far);
CORE_API Boolean CDECL world_set_camera_position(CameraHdl cam, Vec3 pos);
CORE_API Boolean CDECL world_set_camera_direction(CameraHdl cam, Vec3 dir);
CORE_API Boolean CDECL world_set_camera_up(CameraHdl cam, Vec3 up);
CORE_API Boolean CDECL world_set_camera_near(CameraHdl cam, float near);
CORE_API Boolean CDECL world_set_camera_far(CameraHdl cam, float far);
CORE_API Boolean CDECL world_get_pinhole_camera_fov(ConstCameraHdl cam, float* vFov);
CORE_API Boolean CDECL world_set_pinhole_camera_fov(CameraHdl cam, float vFov);
CORE_API Boolean CDECL world_get_focus_camera_focal_length(ConstCameraHdl cam, float* focalLength);
CORE_API Boolean CDECL world_get_focus_camera_focus_distance(ConstCameraHdl cam, float* focusDistance);
CORE_API Boolean CDECL world_get_focus_camera_sensor_height(ConstCameraHdl cam, float* sensorHeight);
CORE_API Boolean CDECL world_get_focus_camera_aperture(ConstCameraHdl cam, float* aperture);
CORE_API Boolean CDECL world_set_focus_camera_focal_length(CameraHdl cam, float focalLength);
CORE_API Boolean CDECL world_set_focus_camera_focus_distance(CameraHdl cam, float focusDistance);
CORE_API Boolean CDECL world_set_focus_camera_sensor_height(CameraHdl cam, float sensorHeight);
CORE_API Boolean CDECL world_set_focus_camera_aperture(CameraHdl cam, float aperture);

// Interface for rendering
CORE_API Boolean CDECL render_enable_renderer(RendererType type);
CORE_API Boolean CDECL render_iterate();
CORE_API Boolean CDECL render_reset();
CORE_API uint32_t CDECL render_get_current_iteration();
// TODO: what do we pass to the GUI?
CORE_API Boolean CDECL render_save_screenshot(const char* filename);
CORE_API Boolean CDECL render_enable_render_target(RenderTarget target, Boolean variance);
CORE_API Boolean CDECL render_disable_render_target(RenderTarget target, Boolean variance);
CORE_API Boolean CDECL render_enable_variance_render_targets();
CORE_API Boolean CDECL render_enable_non_variance_render_targets();
CORE_API Boolean CDECL render_enable_all_render_targets();
CORE_API Boolean CDECL render_disable_variance_render_targets();
CORE_API Boolean CDECL render_disable_non_variance_render_targets();
CORE_API Boolean CDECL render_disable_all_render_targets();
CORE_API uint32_t CDECL render_get_target_opengl_format(RenderTarget target, Boolean variance);
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
CORE_API int32_t CDECL mufflon_get_cuda_device_index();
CORE_API Boolean CDECL mufflon_is_cuda_available();
CORE_API void CDECL mufflon_destroy();

// TODO
CORE_API Boolean CDECL copy_output_to_texture(uint32_t textureId, RenderTarget target, Boolean variance);
CORE_API const char* CDECL core_get_dll_error();
CORE_API Boolean CDECL core_set_log_level(LogLevel level);
CORE_API Boolean CDECL core_set_lod_loader(Boolean (CDECL *func)(ObjectHdl, uint32_t));

}
