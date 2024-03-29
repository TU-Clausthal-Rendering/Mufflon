#pragma once

#include "core_api.h"
#include "texture_data.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <limits.h>

#define INVALID_INDEX int32_t{-1}
#define INVALID_SIZE std::size_t(ULLONG_MAX)
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

typedef struct {
	Vec4 q0;	// i, j, k, r
	Vec4 qe;	// e(i, j, k, r)
} DualQuaternion;

typedef struct {
	uint64_t cycles;
	uint64_t microseconds;
} ProcessTime;

typedef struct {
	uint32_t vertices;
	uint32_t triangles;
	uint32_t quads;
	uint32_t edges;
	uint32_t spheres;
} LodMetadata;

typedef enum {
	CAM_PINHOLE,
	CAM_FOCUS,
	CAM_COUNT
} CameraType;

typedef enum {
	LIGHT_POINT,
	LIGHT_SPOT,
	LIGHT_DIRECTIONAL,
	LIGHT_ENVMAP,
	LIGHT_COUNT
} LightType;

typedef enum {
	BACKGROUND_MONOCHROME,
	BACKGROUND_ENVMAP,
	BACKGROUND_SKY_HOSEK,
	// TODO: Preetham
	BACKGROUND_COUNT
} BackgroundType;

typedef enum {
	SAMPLING_NEAREST,
	SAMPLING_LINEAR
} TextureSampling;

typedef enum {
	SHADOWING_VCAVITY,
	SHADOWING_SMITH
} ShadowingModel;

typedef enum {
	NDF_BECKMANN,
	NDF_GGX,
	NDF_COSINE
} NormalDistFunction;

typedef enum {
	MATERIAL_EMISSIVE,
	MATERIAL_LAMBERT,
	MATERIAL_ORENNAYAR,
	MATERIAL_TORRANCE,
	MATERIAL_WALTER,
	MATERIAL_BLEND,
	MATERIAL_FRESNEL,
	MATERIAL_MICROFACET,
	// MATERIAL_FRESNEL_CONDUCTOR	// Maybe reintroduce
	MATERIAL_NUM,
} MaterialParamType;

typedef enum {
	MEDIUM_NONE,
	MEDIUM_DIELECTRIC,
	MEDIUM_CONDUCTOR
} OuterMediumType;

typedef enum {
	MIPMAP_NONE,
	MIPMAP_AVG,
	MIPMAP_MIN,
	MIPMAP_MAX
} MipmapType;

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

typedef enum {
	ATTRTYPE_CHAR,
	ATTRTYPE_UCHAR,
	ATTRTYPE_SHORT,
	ATTRTYPE_USHORT,
	ATTRTYPE_INT,
	ATTRTYPE_UINT,
	ATTRTYPE_LONG,
	ATTRTYPE_ULONG,
	ATTRTYPE_FLOAT,
	ATTRTYPE_DOUBLE,
	ATTRTYPE_UCHAR2,
	ATTRTYPE_UCHAR3,
	ATTRTYPE_UCHAR4,
	ATTRTYPE_INT2,
	ATTRTYPE_INT3,
	ATTRTYPE_INT4,
	ATTRTYPE_FLOAT2,
	ATTRTYPE_FLOAT3,
	ATTRTYPE_FLOAT4,
	ATTRTYPE_UVEC4,
	ATTRTYPE_SPHERE,
	ATTRTYPE_COUNT
} GeomAttributeType;

typedef struct {
	GeomAttributeType type;
	const char* name;
} VertexAttributeHdl;

typedef struct {
	GeomAttributeType type;
	const char* name;
} FaceAttributeHdl;

typedef struct {
	GeomAttributeType type;
	const char* name;
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
typedef void* MufflonInstanceHdl;
typedef const void* ConstMufflonInstanceHdl;

typedef Vec4(*TextureCallback)(uint32_t x, uint32_t y, uint32_t layer,
							   TextureFormat format, Vec4 value,
							   void* userParams);

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
	ShadowingModel shadowingModel;
	NormalDistFunction ndf;
	TextureHdl albedo;
} TorranceParams;
typedef struct {
	TextureHdl roughness;
	ShadowingModel shadowingModel;
	NormalDistFunction ndf;
	Vec3 absorption;
	float refractionIndex;
} WalterParams;	// Used for Walter AND Microfacet model
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
	float factor;
	struct MaterialParamsStruct* mat;
} BlendLayer;

typedef struct {
	BlendLayer a;
	BlendLayer b;
} BlendParams;
typedef struct {
	Vec2 refractionIndex;
	struct MaterialParamsStruct* a;
	struct MaterialParamsStruct* b;
} FresnelParams;

typedef struct {
	TextureHdl map;
	TextureHdl maxMips;
	float bias;
	float scale;
} MaterialParamsDisplacement;

typedef struct MaterialParamsStruct {
	Medium outerMedium;
	MaterialParamType innerType;
	TextureHdl alpha;
	MaterialParamsDisplacement displacement;
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
typedef enum {
	PARAM_INT,
	PARAM_FLOAT,
	PARAM_BOOL,
	PARAM_ENUM
} ParameterType;


typedef enum {
	BULK_FILE,
	BULK_ARRAY
} BulkType;
// Abstraction for bulk load
typedef struct {
	BulkType type;
	union {
		FILE* file;
		const char* bytes;
	} descriptor;
} BulkLoader;

typedef enum {
	DEVICE_NONE = 0u,
	DEVICE_CPU = 1u,
	DEVICE_CUDA = 2u,
	DEVICE_OPENGL = 4u
} RenderDevice;

// TODO: how to handle errors

// DLL interface
CORE_API const char* CDECL core_get_dll_error();
CORE_API Boolean CDECL core_set_logger(void(*logCallback)(const char*, int));
CORE_API Boolean CDECL core_set_log_level(LogLevel level);

// General mufflon interface
CORE_API MufflonInstanceHdl CDECL mufflon_initialize();
CORE_API void CDECL mufflon_destroy(MufflonInstanceHdl instHdl);
CORE_API Boolean CDECL mufflon_initialize_opengl(MufflonInstanceHdl instHdl);
CORE_API int32_t CDECL mufflon_get_cuda_device_index();
CORE_API Boolean CDECL mufflon_is_cuda_available();
CORE_API Boolean CDECL mufflon_set_lod_loader(MufflonInstanceHdl instHdl, Boolean(*func)(void*, ObjectHdl, uint32_t, Boolean),
											  Boolean(*objFunc)(void*, uint32_t, uint16_t*, uint32_t*),
											  Boolean(*metaFunc)(void*, LodMetadata*, size_t*), void* userParams);

// Render image functions
CORE_API Boolean CDECL mufflon_get_target_image(MufflonInstanceHdl instHdl, const char* name, Boolean variance, const float** ptr);
CORE_API Boolean CDECL mufflon_get_target_image_num_channels(MufflonInstanceHdl instHdl, int* numChannels);
CORE_API Boolean CDECL mufflon_copy_screen_texture_rgba32(MufflonInstanceHdl instHdl, Vec4* ptr, const float factor);
CORE_API Boolean CDECL mufflon_get_pixel_info(MufflonInstanceHdl instHdl, uint32_t x, uint32_t y, Boolean borderClamp, float* r, float* g, float* b, float* a);

// World loading/clearing
CORE_API void CDECL world_clear_all(MufflonInstanceHdl instHdl);
CORE_API Boolean CDECL world_finalize(MufflonInstanceHdl instHdl, const Vec3 min, const Vec3 max, const char** msg);

// Polygon interface
CORE_API Boolean CDECL polygon_reserve(LodHdl lvlDtl, size_t vertices, size_t tris, size_t quads);
CORE_API VertexAttributeHdl CDECL polygon_request_vertex_attribute(LodHdl lvlDtl, const char* name,
																   GeomAttributeType type);
CORE_API FaceAttributeHdl CDECL polygon_request_face_attribute(LodHdl lvlDtl, const char* name,
															   GeomAttributeType type);
CORE_API VertexHdl CDECL polygon_add_vertex(LodHdl lvlDtl, Vec3 point, Vec3 normal, Vec2 uv);
CORE_API FaceHdl CDECL polygon_add_triangle(LodHdl lvlDtl, UVec3 vertices);
CORE_API FaceHdl CDECL polygon_add_triangle_material(LodHdl lvlDtl, UVec3 vertices,
													 MatIdx idx);
CORE_API FaceHdl CDECL polygon_add_quad(LodHdl lvlDtl, UVec4 vertices);
CORE_API FaceHdl CDECL polygon_add_quad_material(LodHdl lvlDtl, UVec4 vertices,
												 MatIdx idx);
CORE_API VertexHdl CDECL polygon_add_vertex_bulk(LodHdl lvlDtl, size_t count, const BulkLoader* points,
												 const BulkLoader* normals, const BulkLoader* uvs,
												 const AABB* aabb, size_t* pointsRead, size_t* normalsRead,
												 size_t* uvsRead);
CORE_API Boolean CDECL polygon_set_vertex_attribute(LodHdl lvlDtl, const VertexAttributeHdl attr,
													VertexHdl vertex, const void* value);
CORE_API Boolean CDECL polygon_set_vertex_normal(LodHdl lvlDtl, VertexHdl vertex, Vec3 normal);
CORE_API Boolean CDECL polygon_set_vertex_uv(LodHdl lvlDtl, VertexHdl vertex, Vec2 uv);
CORE_API Boolean CDECL polygon_set_face_attribute(LodHdl lvlDtl, const FaceAttributeHdl attr,
												  FaceHdl face, const void* value);
CORE_API Boolean CDECL polygon_set_material_idx(LodHdl lvlDtl, FaceHdl face, MatIdx idx);
CORE_API size_t CDECL polygon_set_vertex_attribute_bulk(LodHdl lvlDtl, const VertexAttributeHdl attr,
														VertexHdl startVertex, size_t count,
														const BulkLoader* stream);
CORE_API size_t CDECL polygon_set_face_attribute_bulk(LodHdl lvlDtl, const FaceAttributeHdl attr,
													  FaceHdl startFace, size_t count,
													  const BulkLoader* stream);
CORE_API size_t CDECL polygon_set_material_idx_bulk(LodHdl lvlDtl, FaceHdl startFace, size_t count,
													const BulkLoader* stream);
CORE_API size_t CDECL polygon_get_vertex_count(LodHdl lvlDtl);
CORE_API size_t CDECL polygon_get_face_count(LodHdl lvlDtl);
CORE_API size_t CDECL polygon_get_triangle_count(LodHdl lvlDtl);
CORE_API size_t CDECL polygon_get_quad_count(LodHdl lvlDtl);
CORE_API Boolean CDECL polygon_get_bounding_box(LodHdl lvlDtl, Vec3* min, Vec3* max);

// Sphere interface
CORE_API Boolean CDECL spheres_reserve(LodHdl lvlDtl, size_t count);
CORE_API SphereAttributeHdl CDECL spheres_request_attribute(LodHdl lvlDtl, const char* name,
															GeomAttributeType type);
CORE_API SphereHdl CDECL spheres_add_sphere(LodHdl lvlDtl, Vec3 point, float radius);
CORE_API SphereHdl CDECL spheres_add_sphere_material(LodHdl lvlDtl, Vec3 point, float radius,
													 MatIdx idx);
CORE_API SphereHdl CDECL spheres_add_sphere_bulk(LodHdl lvlDtl, size_t count, const BulkLoader* stream,
												 const AABB* aabb, size_t* readSpheres);
CORE_API Boolean CDECL spheres_set_attribute(LodHdl lvlDtl, const SphereAttributeHdl attr,
											 SphereHdl sphere, const void* value);
CORE_API Boolean CDECL spheres_set_material_idx(LodHdl lvlDtl, SphereHdl sphere, MatIdx idx);
CORE_API size_t CDECL spheres_set_attribute_bulk(LodHdl lvlDtl, const SphereAttributeHdl attr,
												 SphereHdl startSphere, size_t count,
												 const BulkLoader* stream);
CORE_API size_t CDECL spheres_set_material_idx_bulk(LodHdl lvlDtl, SphereHdl startSphere, size_t count,
													const BulkLoader* stream);
CORE_API size_t CDECL spheres_get_sphere_count(LodHdl lvlDtl);
CORE_API Boolean CDECL spheres_get_bounding_box(LodHdl lvlDtl, Vec3* min, Vec3* max);

// Object interface
CORE_API ObjectHdl CDECL world_create_object(MufflonInstanceHdl instHdl, const char* name, ::ObjectFlags flags);
CORE_API ObjectHdl CDECL world_get_object(MufflonInstanceHdl instHdl, const char* name);
CORE_API const char* CDECL world_get_object_name(ObjectHdl obj);
CORE_API Boolean CDECL object_has_lod(ConstObjectHdl obj, LodLevel level);
CORE_API Boolean CDECL object_allocate_lod_slots(ObjectHdl obj, LodLevel slots);
CORE_API LodHdl CDECL object_add_lod(ObjectHdl obj, LodLevel level, Boolean asReduced);
CORE_API Boolean CDECL object_get_id(ObjectHdl obj, uint32_t* id);

// Instance interface
CORE_API void CDECL world_reserve_objects_instances(MufflonInstanceHdl instHdl, const uint32_t objects, const uint32_t instances);
CORE_API InstanceHdl CDECL world_create_instance(MufflonInstanceHdl instHdl, ObjectHdl obj, const uint32_t animationFrame);
CORE_API uint32_t CDECL world_get_instance_count(MufflonInstanceHdl instHdl, const uint32_t frame);
CORE_API InstanceHdl CDECL world_get_instance_by_index(MufflonInstanceHdl instHdl, uint32_t index, const uint32_t animationFrame);
CORE_API Boolean CDECL world_apply_instance_transformation(MufflonInstanceHdl instHdl, InstanceHdl inst);
CORE_API Boolean CDECL instance_set_transformation_matrix(MufflonInstanceHdl instHdl, InstanceHdl inst, const Mat3x4* mat,
														  const Boolean isWorldToInst);
CORE_API Boolean CDECL instance_get_transformation_matrix(MufflonInstanceHdl instHdl, InstanceHdl inst, Mat3x4* mat);
CORE_API Boolean CDECL instance_get_bounding_box(MufflonInstanceHdl instHdl, InstanceHdl inst, Vec3* min, Vec3* max, LodLevel lod);

// Animation interface
CORE_API Boolean CDECL world_set_frame_current(MufflonInstanceHdl instHdl, const uint32_t animationFrame);
CORE_API Boolean CDECL world_get_frame_current(MufflonInstanceHdl instHdl, uint32_t* animationFrame);
CORE_API Boolean CDECL world_get_frame_count(MufflonInstanceHdl instHdl, uint32_t* frameCount);
CORE_API uint32_t CDECL world_get_highest_instance_frame(MufflonInstanceHdl instHdl);
CORE_API void CDECL world_reserve_animation(MufflonInstanceHdl instHdl, const uint32_t numBones, const uint32_t frameCount);
CORE_API void CDECL world_set_bone(MufflonInstanceHdl instHdl, const uint32_t boneIndex, const uint32_t frame,
								   const DualQuaternion* transformation);

// Tessellation interface
CORE_API void CDECL world_set_tessellation_level(MufflonInstanceHdl instHdl, const float maxTessLevel);
CORE_API float CDECL world_get_tessellation_level(MufflonInstanceHdl instHdl);

// Material interface
CORE_API MaterialHdl CDECL world_add_material(MufflonInstanceHdl instHdl, const char* name, const MaterialParams* mat);
CORE_API IndexType CDECL world_get_material_count(MufflonInstanceHdl instHdl);
CORE_API MaterialHdl CDECL world_get_material(MufflonInstanceHdl instHdl, IndexType index);
CORE_API size_t CDECL world_get_material_size(MaterialHdl material);
CORE_API const char* CDECL world_get_material_name(MaterialHdl material);
CORE_API Boolean CDECL world_get_material_data(MufflonInstanceHdl instHdl, MaterialHdl material, MaterialParams* buffer);

// Texture interface
CORE_API TextureHdl CDECL world_get_texture(MufflonInstanceHdl instHdl, const char* path);
CORE_API TextureHdl CDECL world_add_texture(MufflonInstanceHdl instHdl, const char* path, TextureSampling sampling, MipmapType type,
											TextureCallback callback, void* userParams);
CORE_API TextureHdl CDECL world_add_texture_converted(MufflonInstanceHdl instHdl, const char* path, TextureSampling sampling, TextureFormat targetFormat,
													  MipmapType type, TextureCallback callback, void* userParams);
CORE_API TextureHdl CDECL world_add_texture_value(MufflonInstanceHdl instHdl, const float* value, int num, TextureSampling sampling);
CORE_API Boolean CDECL world_add_displacement_map(MufflonInstanceHdl instHdl, const char* path, TextureHdl* hdlTex, TextureHdl* hdlMips);
CORE_API const char* CDECL world_get_texture_name(TextureHdl hdl);
CORE_API Boolean CDECL world_get_texture_size(TextureHdl hdl, IVec2* size);

// Camera interface
CORE_API CameraHdl CDECL world_add_pinhole_camera(MufflonInstanceHdl instHdl, const char* name, const Vec3* position, const Vec3* dir,
												  const Vec3* up, const uint32_t pathCount, float near,
												  float far, float vFov);
CORE_API CameraHdl CDECL world_add_focus_camera(MufflonInstanceHdl instHdl, const char* name, const Vec3* position, const Vec3* dir,
												const Vec3* up, const uint32_t pathCount, float near, float far,
												float focalLength, float focusDistance,
												float lensRad, float chipHeight);
CORE_API Boolean CDECL world_remove_camera(MufflonInstanceHdl instHdl, CameraHdl hdl);
CORE_API size_t CDECL world_get_camera_count(MufflonInstanceHdl instHdl);
CORE_API CameraHdl CDECL world_get_camera(MufflonInstanceHdl instHdl, const char* name);
CORE_API CameraHdl CDECL world_get_camera_by_index(MufflonInstanceHdl instHdl, size_t index);
CORE_API CameraType CDECL world_get_camera_type(ConstCameraHdl cam);
CORE_API const char* CDECL world_get_camera_name(ConstCameraHdl cam);
CORE_API Boolean CDECL world_get_camera_path_segment_count(ConstCameraHdl cam, uint32_t* segments);
CORE_API Boolean CDECL world_get_camera_position(ConstCameraHdl cam, Vec3* pos, const uint32_t pathIndex);
CORE_API Boolean CDECL world_get_camera_current_position(MufflonInstanceHdl instHdl, ConstCameraHdl cam, Vec3* pos);
CORE_API Boolean CDECL world_get_camera_direction(ConstCameraHdl cam, Vec3* dir, const uint32_t pathIndex);
CORE_API Boolean CDECL world_get_camera_current_direction(MufflonInstanceHdl instHdl, ConstCameraHdl cam, Vec3* dir);
CORE_API Boolean CDECL world_get_camera_up(ConstCameraHdl cam, Vec3* up, const uint32_t pathIndex);
CORE_API Boolean CDECL world_get_camera_current_up(MufflonInstanceHdl instHdl, ConstCameraHdl cam, Vec3* up);
CORE_API Boolean CDECL world_get_camera_near(ConstCameraHdl cam, float* near);
CORE_API Boolean CDECL world_get_camera_far(ConstCameraHdl cam, float* far);
CORE_API Boolean CDECL world_set_camera_position(CameraHdl cam, Vec3 pos, const uint32_t pathIndex);
CORE_API Boolean CDECL world_set_camera_current_position(MufflonInstanceHdl instHdl, CameraHdl cam, Vec3 pos);
CORE_API Boolean CDECL world_set_camera_direction(CameraHdl cam, Vec3 dir, Vec3 up, const uint32_t pathIndex);
CORE_API Boolean CDECL world_set_camera_current_direction(MufflonInstanceHdl instHdl, CameraHdl cam, Vec3 dir, Vec3 up);
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

// Scenario interface
CORE_API void CDECL world_reserve_scenarios(MufflonInstanceHdl instHdl, const uint32_t scenarios);
CORE_API ScenarioHdl CDECL world_create_scenario(MufflonInstanceHdl instHdl, const char* name);
CORE_API ScenarioHdl CDECL world_find_scenario(MufflonInstanceHdl instHdl, const char* name);
CORE_API uint32_t CDECL world_get_scenario_count(MufflonInstanceHdl instHdl);
CORE_API ScenarioHdl CDECL world_get_scenario_by_index(MufflonInstanceHdl instHdl, uint32_t index);
CORE_API ConstScenarioHdl CDECL world_get_current_scenario(MufflonInstanceHdl instHdl);
CORE_API Boolean CDECL world_finalize_scenario(MufflonInstanceHdl instHdl, ScenarioHdl scenario, const char** msg);
CORE_API SceneHdl CDECL world_load_scenario(MufflonInstanceHdl instHdl, ScenarioHdl scenario);
CORE_API const char* CDECL scenario_get_name(ScenarioHdl scenario);
CORE_API LodLevel CDECL scenario_get_global_lod_level(ScenarioHdl scenario);
CORE_API Boolean CDECL scenario_set_global_lod_level(ScenarioHdl scenario, LodLevel level);
CORE_API Boolean CDECL scenario_get_resolution(ScenarioHdl scenario, uint32_t* width, uint32_t* height);
CORE_API Boolean CDECL scenario_set_resolution(ScenarioHdl scenario, uint32_t width, uint32_t height);
CORE_API CameraHdl CDECL scenario_get_camera(ScenarioHdl scenario);
CORE_API Boolean CDECL scenario_set_camera(MufflonInstanceHdl instHdl, ScenarioHdl scenario, CameraHdl cam);
CORE_API Boolean CDECL scenario_is_object_masked(ScenarioHdl scenario, ObjectHdl obj);
CORE_API Boolean CDECL scenario_mask_object(ScenarioHdl scenario, ObjectHdl obj);
CORE_API Boolean CDECL scenario_mask_instance(ScenarioHdl scenario, InstanceHdl inst);
CORE_API Boolean CDECL scenario_set_object_tessellation_level(ScenarioHdl scenario, ObjectHdl hdl, float level);
CORE_API Boolean CDECL scenario_set_object_adaptive_tessellation(ScenarioHdl scenario, ObjectHdl hdl, Boolean value);
CORE_API Boolean CDECL scenario_set_object_phong_tessellation(ScenarioHdl scenario, ObjectHdl hdl, Boolean value);
CORE_API Boolean CDECL scenario_has_object_tessellation_info(ScenarioHdl scenario, ObjectHdl hdl, Boolean* value);
CORE_API Boolean CDECL scenario_get_object_tessellation_level(MufflonInstanceHdl instHdl, ScenarioHdl scenario, ObjectHdl hdl, float* level);
CORE_API Boolean CDECL scenario_get_object_adaptive_tessellation(ScenarioHdl scenario, ObjectHdl hdl, Boolean* value);
CORE_API Boolean CDECL scenario_get_object_phong_tessellation(ScenarioHdl scenario, ObjectHdl hdl, Boolean* value);
CORE_API LodLevel CDECL scenario_get_object_lod(ScenarioHdl scenario, ObjectHdl obj);
CORE_API Boolean CDECL scenario_set_object_lod(ScenarioHdl scenario, ObjectHdl obj, LodLevel level);
CORE_API Boolean CDECL scenario_set_instance_lod(ScenarioHdl scenario, InstanceHdl inst, LodLevel level);
CORE_API IndexType CDECL scenario_get_point_light_count(ScenarioHdl scenario);
CORE_API IndexType CDECL scenario_get_spot_light_count(ScenarioHdl scenario);
CORE_API IndexType CDECL scenario_get_dir_light_count(ScenarioHdl scenario);
CORE_API Boolean CDECL scenario_has_envmap_light(MufflonInstanceHdl instHdl, ScenarioHdl scenario);
CORE_API LightHdl CDECL scenario_get_light_handle(ScenarioHdl scenario, IndexType index, LightType type);
CORE_API Boolean CDECL scenario_add_light(MufflonInstanceHdl instHdl, ScenarioHdl scenario, LightHdl hdl);
CORE_API Boolean CDECL scenario_remove_light(MufflonInstanceHdl instHdl, ScenarioHdl scenario, LightHdl hdl);
CORE_API void CDECL scenario_reserve_material_slots(ScenarioHdl scenario, size_t count);
CORE_API void CDECL scenario_reserve_custom_object_properties(ScenarioHdl scenario, size_t objects);
CORE_API void CDECL scenario_reserve_custom_instance_properties(ScenarioHdl scenario, size_t instances);
CORE_API MatIdx CDECL scenario_declare_material_slot(ScenarioHdl scenario, const char* name, size_t nameLength);
CORE_API MatIdx CDECL scenario_get_material_slot(ScenarioHdl scenario, const char* name, size_t nameLength);
CORE_API const char* scenario_get_material_slot_name(ScenarioHdl scenario, MatIdx slot);
CORE_API size_t CDECL scenario_get_material_slot_count(ScenarioHdl scenario);
CORE_API MaterialHdl CDECL scenario_get_assigned_material(ScenarioHdl scenario, MatIdx index);
CORE_API Boolean CDECL scenario_assign_material(ScenarioHdl scenario, MatIdx index, MaterialHdl handle);

// Scene interface
CORE_API SceneHdl CDECL world_get_current_scene(MufflonInstanceHdl instHdl);
CORE_API Boolean CDECL scene_get_bounding_box(SceneHdl scene, Vec3* min, Vec3* max);
CORE_API ConstCameraHdl CDECL scene_get_camera(SceneHdl scene);
CORE_API Boolean CDECL scene_move_active_camera(MufflonInstanceHdl instHdl, float x, float y, float z);
CORE_API Boolean CDECL scene_rotate_active_camera(MufflonInstanceHdl instHdl, float x, float y, float z);
CORE_API Boolean CDECL scene_is_sane(MufflonInstanceHdl instHdl);
CORE_API Boolean CDECL scene_request_retessellation(MufflonInstanceHdl instHdl);

// Light interface
CORE_API LightHdl CDECL world_add_light(MufflonInstanceHdl instHdl, const char* name, LightType type, const uint32_t count);
CORE_API LightHdl CDECL world_add_background_light(MufflonInstanceHdl instHdl, const char* name, BackgroundType type);
CORE_API Boolean CDECL world_set_light_name(MufflonInstanceHdl instHdl, LightHdl hdl, const char* newName);
CORE_API Boolean CDECL world_remove_light(MufflonInstanceHdl instHdl, LightHdl hdl);
CORE_API Boolean CDECL world_find_light(MufflonInstanceHdl instHdl, const char* name, LightHdl* hdl);
CORE_API size_t CDECL world_get_point_light_count(MufflonInstanceHdl instHdl);
CORE_API size_t CDECL world_get_spot_light_count(MufflonInstanceHdl instHdl);
CORE_API size_t CDECL world_get_dir_light_count(MufflonInstanceHdl instHdl);
CORE_API size_t CDECL world_get_env_light_count(MufflonInstanceHdl instHdl);
CORE_API LightHdl CDECL world_get_light_handle(size_t index, LightType type);
CORE_API LightType CDECL world_get_light_type(LightHdl hdl);
CORE_API BackgroundType CDECL world_get_env_light_type(MufflonInstanceHdl instHdl, LightHdl hdl);
CORE_API const char* CDECL world_get_light_name(MufflonInstanceHdl instHdl, LightHdl hdl);
CORE_API Boolean CDECL world_get_point_light_position(MufflonInstanceHdl instHdl, ConstLightHdl hdl, Vec3* pos, const uint32_t frame);
CORE_API Boolean CDECL world_get_point_light_intensity(MufflonInstanceHdl instHdl, ConstLightHdl hdl, Vec3* intensity, const uint32_t frame);
CORE_API Boolean CDECL world_get_point_light_path_segments(MufflonInstanceHdl instHdl, ConstLightHdl hdl, uint32_t* segments);
CORE_API Boolean CDECL world_set_point_light_position(MufflonInstanceHdl instHdl, LightHdl hdl, Vec3 pos, const uint32_t frame);
CORE_API Boolean CDECL world_set_point_light_intensity(MufflonInstanceHdl instHdl, LightHdl hdl, Vec3 intensity, const uint32_t frame);
CORE_API Boolean CDECL world_get_spot_light_path_segments(MufflonInstanceHdl instHdl, ConstLightHdl hdl, uint32_t* segments);
CORE_API Boolean CDECL world_get_spot_light_position(MufflonInstanceHdl instHdl, ConstLightHdl hdl, Vec3* pos, const uint32_t frame);
CORE_API Boolean CDECL world_get_spot_light_intensity(MufflonInstanceHdl instHdl, ConstLightHdl hdl, Vec3* intensity, const uint32_t frame);
CORE_API Boolean CDECL world_get_spot_light_direction(MufflonInstanceHdl instHdl, ConstLightHdl hdl, Vec3* direction, const uint32_t frame);
CORE_API Boolean CDECL world_get_spot_light_angle(MufflonInstanceHdl instHdl, ConstLightHdl hdl, float* angle, const uint32_t frame);
CORE_API Boolean CDECL world_get_spot_light_falloff(MufflonInstanceHdl instHdl, ConstLightHdl hdl, float* falloff, const uint32_t frame);
CORE_API Boolean CDECL world_set_spot_light_position(MufflonInstanceHdl instHdl, LightHdl hdl, Vec3 pos, const uint32_t frame);
CORE_API Boolean CDECL world_set_spot_light_intensity(MufflonInstanceHdl instHdl, LightHdl hdl, Vec3 intensity, const uint32_t frame);
CORE_API Boolean CDECL world_set_spot_light_direction(MufflonInstanceHdl instHdl, LightHdl hdl, Vec3 direction, const uint32_t frame);
CORE_API Boolean CDECL world_set_spot_light_angle(MufflonInstanceHdl instHdl, LightHdl hdl, float angle, const uint32_t frame);
CORE_API Boolean CDECL world_set_spot_light_falloff(MufflonInstanceHdl instHdl, LightHdl hdl, float falloff, const uint32_t frame);
CORE_API Boolean CDECL world_get_dir_light_path_segments(MufflonInstanceHdl instHdl, ConstLightHdl hdl, uint32_t* segments);
CORE_API Boolean CDECL world_get_dir_light_direction(MufflonInstanceHdl instHdl, ConstLightHdl hdl, Vec3* direction, const uint32_t frame);
CORE_API Boolean CDECL world_get_dir_light_irradiance(MufflonInstanceHdl instHdl, ConstLightHdl hdl, Vec3* irradiance, const uint32_t frame);
CORE_API Boolean CDECL world_set_dir_light_direction(MufflonInstanceHdl instHdl, LightHdl hdl, Vec3 direction, const uint32_t frame);
CORE_API Boolean CDECL world_set_dir_light_irradiance(MufflonInstanceHdl instHdl, LightHdl hdl, Vec3 irradiance, const uint32_t frame);
CORE_API const char* CDECL world_get_env_light_map(MufflonInstanceHdl instHdl, ConstLightHdl hdl);
CORE_API Boolean CDECL world_get_env_light_scale(MufflonInstanceHdl instHdl, LightHdl hdl, Vec3* color);
CORE_API Boolean CDECL world_get_sky_light_turbidity(MufflonInstanceHdl instHdl, LightHdl hdl, float* turbidity);
CORE_API Boolean CDECL world_set_sky_light_turbidity(MufflonInstanceHdl instHdl, LightHdl hdl, float turbidity);
CORE_API Boolean CDECL world_get_sky_light_albedo(MufflonInstanceHdl instHdl, LightHdl hdl, float* albedo);
CORE_API Boolean CDECL world_set_sky_light_albedo(MufflonInstanceHdl instHdl, LightHdl hdl, float albedo);
CORE_API Boolean CDECL world_get_sky_light_solar_radius(MufflonInstanceHdl instHdl, LightHdl hdl, float* radius);
CORE_API Boolean CDECL world_set_sky_light_solar_radius(MufflonInstanceHdl instHdl, LightHdl hdl, float radius);
CORE_API Boolean CDECL world_get_sky_light_sun_direction(MufflonInstanceHdl instHdl, LightHdl hdl, Vec3* sunDir);
CORE_API Boolean CDECL world_set_sky_light_sun_direction(MufflonInstanceHdl instHdl, LightHdl hdl, Vec3 sunDir);
CORE_API Boolean CDECL world_get_env_light_color(MufflonInstanceHdl instHdl, LightHdl hdl, Vec3* color);
CORE_API Boolean CDECL world_set_env_light_color(MufflonInstanceHdl instHdl, LightHdl hdl, Vec3 color);
CORE_API Boolean CDECL world_set_env_light_map(MufflonInstanceHdl instHdl, LightHdl hdl, TextureHdl tex);
CORE_API Boolean CDECL world_set_env_light_scale(MufflonInstanceHdl instHdl, LightHdl hdl, Vec3 color);

// Render interfaces
CORE_API uint32_t CDECL render_get_renderer_count(MufflonInstanceHdl instHdl);
CORE_API uint32_t CDECL render_get_renderer_variations(MufflonInstanceHdl instHdl, uint32_t index);
CORE_API const char* CDECL render_get_renderer_name(MufflonInstanceHdl instHdl, uint32_t index);
CORE_API const char* CDECL render_get_renderer_short_name(MufflonInstanceHdl instHdl, uint32_t index);
CORE_API RenderDevice CDECL render_get_renderer_devices(MufflonInstanceHdl instHdl, uint32_t index, uint32_t variation);
CORE_API Boolean CDECL render_enable_renderer(MufflonInstanceHdl instHdl, uint32_t index, uint32_t variation);
CORE_API Boolean CDECL render_iterate(MufflonInstanceHdl instHdl, ProcessTime* time, ProcessTime* preTime, ProcessTime* postTime);
CORE_API uint32_t CDECL render_get_current_iteration(ConstMufflonInstanceHdl instHdl);
CORE_API Boolean CDECL render_reset(MufflonInstanceHdl instHdl);
CORE_API Boolean CDECL render_save_screenshot(MufflonInstanceHdl instHdl, const char* filename, const char* targetName, Boolean variance);
CORE_API Boolean CDECL render_save_denoised_radiance(MufflonInstanceHdl instHdl, const char* filename);
CORE_API uint32_t CDECL render_get_render_target_count(MufflonInstanceHdl instHdl);
CORE_API const char* CDECL render_get_render_target_name(MufflonInstanceHdl instHdl, uint32_t index);
CORE_API Boolean CDECL render_enable_render_target(MufflonInstanceHdl instHdl, const char* target, Boolean variance);
CORE_API Boolean CDECL render_disable_render_target(MufflonInstanceHdl instHdl, const char* target, Boolean variance);
CORE_API Boolean CDECL render_is_render_target_enabled(MufflonInstanceHdl instHdl, const char* name, Boolean variance);
CORE_API Boolean CDECL render_is_render_target_required(MufflonInstanceHdl instHdl, const char* name, Boolean variance);

// Renderer parameter interface
CORE_API uint32_t CDECL renderer_get_num_parameters(MufflonInstanceHdl instHdl);
CORE_API const char* CDECL renderer_get_parameter_desc(MufflonInstanceHdl instHdl, uint32_t idx, ParameterType* type);
CORE_API Boolean CDECL renderer_set_parameter_int(MufflonInstanceHdl instHdl, const char* name, int32_t value);
CORE_API Boolean CDECL renderer_get_parameter_int(MufflonInstanceHdl instHdl, const char* name, int32_t* value);
CORE_API Boolean CDECL renderer_set_parameter_float(MufflonInstanceHdl instHdl, const char* name, float value);
CORE_API Boolean CDECL renderer_get_parameter_float(MufflonInstanceHdl instHdl, const char* name, float* value);
CORE_API Boolean CDECL renderer_set_parameter_bool(MufflonInstanceHdl instHdl, const char* name, Boolean value);
CORE_API Boolean CDECL renderer_get_parameter_bool(MufflonInstanceHdl instHdl, const char* name, Boolean* value);
CORE_API Boolean CDECL renderer_set_parameter_enum(MufflonInstanceHdl instHdl, const char* name, int value);
CORE_API Boolean CDECL renderer_get_parameter_enum(MufflonInstanceHdl instHdl, const char* name, int* value);
CORE_API Boolean CDECL renderer_get_parameter_enum_count(MufflonInstanceHdl instHdl, const char* param, uint32_t* count);
CORE_API Boolean CDECL renderer_get_parameter_enum_value_from_index(MufflonInstanceHdl instHdl, const char* param, uint32_t index, int* value);
CORE_API Boolean CDECL renderer_get_parameter_enum_value_from_name(MufflonInstanceHdl instHdl, const char* param, const char* valueName, int* value);
CORE_API Boolean CDECL renderer_get_parameter_enum_index_from_value(MufflonInstanceHdl instHdl, const char* param, int value, uint32_t* index);
CORE_API Boolean CDECL renderer_get_parameter_enum_name(MufflonInstanceHdl instHdl, const char* param, int value, const char** name);

// Profiling interface
CORE_API void CDECL profiling_enable();
CORE_API void CDECL profiling_disable();
CORE_API Boolean CDECL profiling_set_level(ProfilingLevel level);
CORE_API Boolean CDECL profiling_save_current_state(const char* path);
CORE_API Boolean CDECL profiling_save_snapshots(const char* path);
CORE_API Boolean CDECL profiling_save_total_and_snapshots(const char* path);
CORE_API const char* CDECL profiling_get_current_state();
CORE_API const char* CDECL profiling_get_snapshots();
CORE_API const char* CDECL profiling_get_total();
CORE_API const char* CDECL profiling_get_total_and_snapshots();
CORE_API void CDECL profiling_reset();
CORE_API size_t CDECL profiling_get_total_cpu_memory();
CORE_API size_t CDECL profiling_get_free_cpu_memory();
CORE_API size_t CDECL profiling_get_used_cpu_memory();
CORE_API size_t CDECL profiling_get_total_gpu_memory();
CORE_API size_t CDECL profiling_get_free_gpu_memory();
CORE_API size_t CDECL profiling_get_used_gpu_memory();

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
