import ctypes
from ctypes import *
from time import *
from enum import IntEnum
import ntpath
import os

class ProcessTime(Structure):
    _fields_ = [
        ("cycles", c_ulonglong),
        ("microseconds", c_ulonglong)
    ]
    
class Vec2(Structure):
    _fields_ = [
        ("u", c_float),
        ("v", c_float)
    ]
    
class Vec3(Structure):
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("z", c_float)
    ]
    
class UVec3(Structure):
    _fields_ = [
        ("x", c_uint32),
        ("y", c_uint32),
        ("z", c_uint32)
    ]
    
class UVec4(Structure):
    _fields_ = [
        ("x", c_uint32),
        ("y", c_uint32),
        ("z", c_uint32),
        ("w", c_uint32)
    ]

class NormalDistFunction(IntEnum):
    BECKMANN = 0,
    GGX = 1,
    COSINE = 2

class ShadowingModel(IntEnum):
    VCAVITY = 0,
    SMITH = 1
    
class MaterialParamType(IntEnum):
    EMISSIVE = 0,
    LAMBERT = 1,
    ORENNAYAR = 2,
    TORRANCE = 3,
    WALTER = 4,
    BLEND = 5,
    FRESNEL = 6,
    MICROFACET = 7

class TextureSampling(IntEnum):
    NEAREST = 0,
    LINEAR = 1

class MipmapType(IntEnum):
    NONE = 0,
    AVERAGE = 1,
    MIN = 2,
    MAX = 3
    
class Medium(Structure):
    _fields_ = [
        ("refractionIndex", Vec2),
        ("absorption", Vec3)
    ]
    
class LambertParams(Structure):
    _fields_ = [
        ("albedo", c_void_p)
    ]
class TorranceParams(Structure):
    _fields_ = [
        ("roughness", c_void_p),
        ("shadowingModel", c_uint32),
        ("ndf", c_uint32),
        ("albedo", c_void_p)
    ]
class WalterParams(Structure):
    _fields_ = [
        ("roughness", c_void_p),
        ("shadowingModel", c_uint32),
        ("ndf", c_uint32),
        ("absorption", Vec3),
        ("refractionIndex", c_float)
    ]
class EmissiveParams(Structure):
    _fields_ = [
        ("radiance", c_void_p),
        ("scale", Vec3)
    ]
class OrennayarParams(Structure):
    _fields_ = [
        ("albedo", c_void_p),
        ("roughness", c_float)
    ]
class MaterialParams(Structure):
    pass
class BlendLayer(Structure):
    _fields_ = [
        ("factor", c_float),
        ("mat", POINTER(MaterialParams))
    ]
class BlendParams(Structure):
    _fields_ = [
        ("a", BlendLayer),
        ("b", BlendLayer)
    ]
class FresnelParams(Structure):
    _fields_ = [
        ("refractionIndex", Vec2),
        ("a", POINTER(MaterialParams)),
        ("b", POINTER(MaterialParams))
    ]
class MaterialParamsDisplacement(Structure):
    _fields_ = [
        ("map", c_void_p),
        ("maxMips", c_void_p),
        ("bias", c_float),
        ("scale", c_float)
    ]
class MaterialUnion(Union):
    _fields_ = [
        ("lambert", LambertParams),
        ("torrance", TorranceParams),
        ("walter", WalterParams),
        ("emissive", EmissiveParams),
        ("orennayar", OrennayarParams),
        ("blend", BlendParams),
        ("fresnel", FresnelParams)
    ]

MaterialParams._fields_ = [
    ("outerMedium", Medium),
    ("innerType", c_uint32),
    ("alpha", c_void_p),
    ("displacement", MaterialParamsDisplacement),
    ("inner", MaterialUnion)
]

class Device(IntEnum):
    CPU = 1,
    CUDA = 2,
    OPENGL = 4

class LogLevel(IntEnum):
    PEDANTIC = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL_ERROR = 4
    
class LoaderStatus(IntEnum):
    SUCCESS = 0,
    ERROR = 1,
    ABORT = 2
    
class LightType(IntEnum):
    POINT = 0,
    SPOT = 1,
    DIRECTIONAL = 2,
    ENVMAP = 3

class DllHolder:
    # Incomplete interface, but enough for Blender integration (currently)
    def __init__(self, binary_path, useLoader):
        # We have to load all DLLs that core references because the
        # (non-existing) rpath isn't set properly
        if binary_path:
            dir = binary_path
        else:
            dir = os.path.dirname(os.path.realpath(__file__))
        self.openmeshcore = ctypes.CDLL(dir + "/OpenMeshCore.dll",mode=ctypes.RTLD_GLOBAL)
        self.openmeshtools = ctypes.CDLL(dir + "/OpenMeshTools.dll",mode=ctypes.RTLD_GLOBAL)
        self.tbb = ctypes.CDLL(dir + "/tbb.dll",mode=ctypes.RTLD_GLOBAL)
        self.openimagedenoise = ctypes.CDLL(dir + "/OpenImageDenoise.dll",mode=ctypes.RTLD_GLOBAL)
        self.core = ctypes.CDLL(dir + "/core.dll",mode=ctypes.RTLD_GLOBAL)
        
        self.core.mufflon_initialize.restype = c_void_p
        self.core.mufflon_destroy.argtypes = [c_void_p]
        self.core.core_get_dll_error.restype = c_char_p
        self.core.world_clear_all.argtypes = [c_void_p]
        self.core.render_get_renderer_name.restype = c_char_p
        self.core.render_get_renderer_name.argtypes = [c_void_p, c_uint32]
        self.core.render_get_renderer_short_name.restype = c_char_p
        self.core.render_get_renderer_short_name.argtypes = [c_void_p, c_uint32]
        self.core.render_get_render_target_name.restype = c_char_p
        self.core.render_get_render_target_name.argtypes = [c_void_p, c_uint32]
        self.core.render_get_renderer_count.restype = c_uint32
        self.core.render_get_renderer_count.argtypes = [c_void_p]
        self.core.render_get_renderer_variations.restype = c_uint32
        self.core.render_get_renderer_variations.argtypes = [c_void_p, c_uint32]
        self.core.render_get_renderer_devices.restype = c_uint
        self.core.render_get_renderer_devices.argtypes = [c_void_p, c_uint32, c_uint32]
        self.core.renderer_set_parameter_int.restype = c_bool
        self.core.renderer_set_parameter_int.argtypes = [c_void_p, c_char_p, c_int32]
        self.core.renderer_get_parameter_int.restype = c_bool
        self.core.renderer_get_parameter_int.argtypes = [c_void_p, c_char_p, POINTER(c_int32)]
        self.core.renderer_set_parameter_float.restype = c_bool
        self.core.renderer_set_parameter_float.argtypes = [c_void_p, c_char_p, c_float]
        self.core.renderer_get_parameter_float.restype = c_bool
        self.core.renderer_get_parameter_float.argtypes = [c_void_p, c_char_p, POINTER(c_float)]
        self.core.renderer_set_parameter_enum.restype = c_bool
        self.core.renderer_set_parameter_enum.argtypes = [c_void_p, c_char_p, c_int32]
        self.core.renderer_get_parameter_enum_value_from_name.restype = c_bool
        self.core.renderer_get_parameter_enum_value_from_name.argtypes = [c_void_p, c_char_p, c_char_p, c_void_p]
        self.core.render_enable_renderer.restype = c_bool
        self.core.render_enable_renderer.argtypes = [c_void_p, c_uint32, c_uint32]
        self.core.render_enable_render_target.argtypes = [c_void_p, c_char_p, c_bool]
        self.core.render_enable_render_target.argtypes = [c_void_p, c_char_p, c_bool]
        self.core.render_disable_render_target.argtypes = [c_void_p, c_char_p, c_bool]
        self.core.render_disable_render_target.argtypes = [c_void_p, c_char_p, c_bool]
        self.core.render_is_render_target_enabled.argtypes = [c_void_p, c_char_p, c_bool]
        self.core.render_iterate.argtypes = [c_void_p, POINTER(ProcessTime)]
        self.core.render_get_current_iteration.restype = c_uint32
        self.core.render_get_current_iteration.argtypes = [c_void_p]
        self.core.scenario_get_name.restype = c_char_p
        self.core.scenario_get_name.argtypes = [c_void_p]
        self.core.world_set_frame_current.argtypes = [c_void_p, c_uint]
        self.core.world_get_current_scenario.restype = c_void_p
        self.core.world_find_scenario.restype = c_void_p
        self.core.world_find_scenario.argtypes = [c_void_p, c_char_p]
        self.core.world_load_scenario.restype = c_void_p
        self.core.world_load_scenario.argtypes = [c_void_p, c_void_p]
        self.core.world_get_frame_current.restype = c_uint32
        self.core.world_get_frame_current.argtypes = [c_void_p, POINTER(c_uint32)]
        self.core.world_set_frame_current.restype = c_uint32
        self.core.world_set_frame_current.argtypes = [c_void_p, c_uint32]
        self.core.world_get_frame_count.restype = c_uint32
        self.core.world_get_frame_count.argtypes = [c_void_p, POINTER(c_uint32)]
        self.core.render_get_render_target_count.restype = c_uint32
        self.core.render_get_render_target_count.argtypes = [c_void_p]
        self.core.mufflon_get_target_image.restype = c_bool
        self.core.mufflon_get_target_image.argtypes = [c_void_p, c_char_p, c_uint32, POINTER(POINTER(c_float))]
        self.core.mufflon_copy_screen_texture_rgba32.restype = c_bool
        self.core.mufflon_copy_screen_texture_rgba32.argtypes = [c_void_p, POINTER(c_float), c_float] 
        self.core.render_save_screenshot.argtypes = [c_void_p, c_char_p, c_char_p, c_bool]
        self.core.render_save_denoised_radiance.argtypes = [c_void_p, c_char_p]
        self.core.world_reserve_objects_instances.restype = c_bool
        self.core.world_reserve_objects_instances.argtypes = [c_void_p, c_size_t, c_size_t]
        self.core.world_create_object.restype = c_void_p
        self.core.world_create_object.argtypes = [c_void_p, c_char_p, c_int]
        self.core.object_add_lod.restype = c_void_p
        self.core.object_add_lod.argtypes = [c_void_p, c_uint]
        self.core.polygon_reserve.argtypes = [c_void_p, c_size_t, c_size_t, c_size_t, c_size_t]
        self.core.polygon_add_vertex.restype = c_int
        self.core.polygon_add_vertex.argtypes = [c_void_p, Vec3, Vec3, Vec2]
        self.core.polygon_add_triangle_material.restype = c_int
        self.core.polygon_add_triangle_material.argtypes = [c_void_p, UVec3, c_ushort]
        self.core.polygon_add_quad_material.restype = c_int
        self.core.polygon_add_quad_material.argtypes = [c_void_p, UVec4, c_ushort]
        self.core.world_create_instance.restype = c_void_p
        self.core.world_create_instance.argtypes = [c_void_p, c_void_p, c_uint32]
        self.core.world_add_pinhole_camera.restype = c_void_p
        self.core.world_add_pinhole_camera.argtypes = [c_void_p, c_char_p, POINTER(Vec3), POINTER(Vec3), POINTER(Vec3),
                                                                 c_uint, c_float, c_float, c_float]
        self.core.world_add_focus_camera.restype = c_void_p
        self.core.world_add_focus_camera.argtypes = [c_void_p, c_char_p, POINTER(Vec3), POINTER(Vec3), POINTER(Vec3),
                                                     c_uint32, c_float, c_float, c_float, c_float, c_float, c_float]
        self.core.world_add_light.restype = c_uint32
        self.core.world_add_light.argtypes = [c_void_p, c_char_p, c_uint32, c_uint]
        self.core.world_set_point_light_position.argtypes = [c_void_p, c_uint32, Vec3, c_uint]
        self.core.world_set_point_light_intensity.argtypes = [c_void_p, c_uint32, Vec3, c_uint]
        self.core.world_set_spot_light_position.argtypes = [c_void_p, c_uint32, Vec3, c_uint]
        self.core.world_set_spot_light_intensity.argtypes = [c_void_p, c_uint32, Vec3, c_uint]
        self.core.world_set_spot_light_direction.argtypes = [c_void_p, c_uint32, Vec3, c_uint]
        self.core.world_set_spot_light_angle.argtypes = [c_void_p, c_uint32, c_float, c_uint]
        self.core.world_set_spot_light_falloff.argtypes = [c_void_p, c_uint32, c_float, c_uint]
        self.core.world_set_dir_light_direction.argtypes = [c_void_p, c_uint32, Vec3, c_uint]
        self.core.world_set_dir_light_irradiance.argtypes = [c_void_p, c_uint32, Vec3, c_uint]
        self.core.world_add_texture.restype = c_void_p
        self.core.world_add_texture.argtypes = [c_void_p, c_char_p, c_uint32, c_uint32, c_void_p, c_void_p]
        self.core.world_add_texture_value.restype = c_void_p
        self.core.world_add_texture_value.argtypes = [c_void_p, POINTER(c_float), c_int32, c_uint32]
        self.core.world_add_material.restype = c_void_p
        self.core.world_add_material.argtypes = [c_void_p, c_char_p, POINTER(MaterialParams)]
        self.core.world_finalize.restype = c_bool
        self.core.world_finalize.argtypes = [c_void_p, POINTER(c_char_p)]
        self.core.world_reserve_scenarios.argtypes = [c_void_p, c_uint32]
        self.core.world_create_scenario.restype = c_void_p
        self.core.world_create_scenario.argtypes = [c_void_p, c_char_p]
        self.core.scenario_set_camera.argtypes = [c_void_p, c_void_p, c_void_p]
        self.core.scenario_set_resolution.argtypes = [c_void_p, c_uint, c_uint]
        self.core.scenario_add_light.argtypes = [c_void_p, c_void_p, c_uint32]
        self.core.scenario_reserve_material_slots.argtypes = [c_void_p, c_size_t]
        self.core.scenario_declare_material_slot.restype = c_ushort
        self.core.scenario_declare_material_slot.argtypes = [c_void_p, c_char_p, c_size_t]
        self.core.scenario_assign_material.restype = c_bool
        self.core.scenario_assign_material.argtypes = [c_void_p, c_ushort, c_void_p]
        self.core.world_finalize_scenario.restype = c_bool
        self.core.world_finalize_scenario.argtypes = [c_void_p, c_void_p, POINTER(c_char_p)]
        self.core.world_load_scenario.argtypes = [c_void_p, c_void_p]
        self.core.instance_set_transformation_matrix.restype = c_bool
        self.core.instance_set_transformation_matrix.argtypes = [c_void_p, c_void_p, POINTER(c_float), c_uint32]
        self.core.world_set_camera_position.restype = c_bool
        self.core.world_set_camera_position.argtypes = [c_void_p, Vec3, c_uint32]
        self.core.world_set_camera_direction.restype = c_bool
        self.core.world_set_camera_direction.argtypes = [c_void_p, Vec3, Vec3, c_uint32]
        self.core.world_set_pinhole_camera_fov.restype = c_bool
        self.core.world_set_pinhole_camera_fov.argtypes = [c_void_p, c_float]
        
        if useLoader:
            self.mffLoader = ctypes.CDLL(dir + "/mffloader.dll",mode=ctypes.RTLD_GLOBAL)
            self.mffLoader.loader_initialize.restype = c_void_p
            self.mffLoader.loader_initialize.argtypes = [c_void_p]
            self.mffLoader.loader_load_json.restype = LoaderStatus
            self.mffLoader.loader_load_json.argtypes = [c_void_p, c_char_p]
        else:
            self.mffLoader = None

class DllInterface:
    def __init__(self, binary_path, useLoader):
        self.dllHolder = DllHolder(binary_path, useLoader)
        self.muffInst = self.dllHolder.core.mufflon_initialize()
        if useLoader:
            self.muffLoaderInst = self.dllHolder.mffLoader.loader_initialize(self.muffInst)
        self.disable_profiling()
        
    def __del__(self):
        self.dllHolder.core.mufflon_destroy(self.muffInst)
        
    def core_get_dll_error(self):
        return self.dllHolder.core.core_get_dll_error().decode()
    
    def core_set_log_level(self, logLevel):
        return self.dllHolder.core.core_set_log_level(c_int32(logLevel)) != 0

    def disable_profiling(self):
        self.dllHolder.core.profiling_disable()
        if self.dllHolder.mffLoader is not None:
            self.dllHolder.mffLoader.loader_profiling_disable()
        
    def world_clear_all(self):
        self.dllHolder.core.world_clear_all(self.muffInst)

    def loader_load_json(self, sceneJson):
        return self.dllHolder.mffLoader.loader_load_json(self.muffLoaderInst, c_char_p(sceneJson.encode('utf-8')))

    def renderer_set_parameter_bool(self, parameterName, value):
        return self.dllHolder.core.renderer_set_parameter_bool(self.muffInst, c_char_p(parameterName.encode('utf-8')), c_bool(value))

    def renderer_set_parameter_float(self, parameterName, value):
        return self.dllHolder.core.renderer_set_parameter_float(self.muffInst, c_char_p(parameterName.encode('utf-8')), c_float(value))

    def renderer_set_parameter_int(self, parameterName, value):
        return self.dllHolder.core.renderer_set_parameter_int(self.muffInst, c_char_p(parameterName.encode('utf-8')), c_int32(value))
    
    def renderer_set_parameter_enum(self, parameterName, value):
        return self.dllHolder.core.renderer_set_parameter_enum(self.muffInst, c_char_p(parameterName.encode('utf-8')), c_int32(value))

    def renderer_get_parameter_enum_value(self, parameterName, valueName):
        value = c_int(0)
        if not self.dllHolder.core.renderer_get_parameter_enum_value_from_name(self.muffInst, c_char_p(parameterName.encode('utf-8')), c_char_p(valueName.encode('utf-8')), byref(value)):
            raise Exception("Failed to retrieve enum parameter '" + parameterName + "' value '" + valueName + "'")
        return value.value

    def renderer_get_parameter_enum_count(self, parameterName):
        count = 0
        if not self.dllHolder.core.renderer_get_parameter_enum_count(self.muffInst, c_char_p(parameterName.encode('utf-8')), byref(count)):
            raise Exception("Failed to retrieve enum parameter '" + parameterName + "' count")
        return count

    def render_iterate(self):
        iterateTime = ProcessTime(0,0)
        preTime = ProcessTime(0,0)
        postTime = ProcessTime(0,0)
        self.dllHolder.core.render_iterate(self.muffInst, byref(iterateTime))
        return iterateTime, preTime, postTime

    def render_reset(self):
        return self.dllHolder.core.render_reset(self.muffInst)

    def render_get_current_iteration(self):
        return self.dllHolder.core.render_get_current_iteration(self.muffInst)

    def render_save_screenshot(self, fileName, targetName, variance):
        return self.dllHolder.core.render_save_screenshot(self.muffInst, c_char_p(fileName.encode('utf-8')), c_char_p(targetName.encode('utf-8')), c_uint32(variance))
    
    def render_save_denoised_radiance(self, fileName):
        return self.dllHolder.core.render_save_denoised_radiance(self.muffInst, c_char_p(fileName.encode('utf-8')))
        
    def render_get_render_target_count(self):
        return self.dllHolder.core.render_get_render_target_count(self.muffInst)

    def render_get_render_target_name(self, index):
        return self.dllHolder.core.render_get_render_target_name(self.muffInst, c_uint32(index)).decode()

    def render_is_render_target_enabled(self, targetName, variance):
        return self.dllHolder.core.render_is_render_target_enabled(self.muffInst, c_char_p(targetName.encode('utf-8')), c_bool(variance))

    def render_enable_render_target(self, targetName, variance):
        return self.dllHolder.core.render_enable_render_target(self.muffInst, c_char_p(targetName.encode('utf-8')), c_bool(variance))
    
    def render_disable_render_target(self, targetName, variance):
        return self.dllHolder.core.render_disable_render_target(self.muffInst, c_char_p(targetName.encode('utf-8')), c_bool(variance))

    def render_enable_renderer(self, rendererIndex, variation):
        return self.dllHolder.core.render_enable_renderer(self.muffInst, c_uint32(rendererIndex), c_uint32(variation))

    def render_get_renderer_count(self):
        return self.dllHolder.core.render_get_renderer_count(self.muffInst)

    def render_get_renderer_variations(self, index):
        return self.dllHolder.core.render_get_renderer_variations(self.muffInst, c_uint32(index))

    def render_get_renderer_name(self, rendererIndex):
        return self.dllHolder.core.render_get_renderer_name(self.muffInst, c_uint32(rendererIndex)).decode()

    def render_get_renderer_short_name(self, rendererIndex):
        return self.dllHolder.core.render_get_renderer_short_name(self.muffInst, c_uint32(rendererIndex)).decode()

    def render_get_renderer_devices(self, rendererIndex, variation):
        return self.dllHolder.core.render_get_renderer_devices(self.muffInst, c_uint32(rendererIndex), c_uint32(variation))

    def render_get_active_scenario_name(self):
        return self.dllHolder.core.scenario_get_name(self.dllHolder.core.world_get_current_scenario()).decode()

    def world_set_frame_current(self, frame):
        return self.dllHolder.core.world_set_frame_current(self.muffInst, c_uint(frame))
    
    def world_get_frame_current(self, frame):
        return self.dllHolder.core.world_get_frame_current(self.muffInst, byref(frame))
    
    def world_get_frame_count(self, frame):
        return self.dllHolder.core.world_get_frame_count(self.muffInst, byref(frame))

    def world_find_scenario(self, name):
        return self.dllHolder.core.world_find_scenario(self.muffInst, c_char_p(name.encode('utf-8')))

    def world_load_scenario(self, hdl):
        return self.dllHolder.core.world_load_scenario(self.muffInst, c_void_p(hdl))

    def world_get_current_scenario(self):
        return self.dllHolder.core.world_get_current_scenario(self.muffInst)
    
    def world_set_tessellation_level(self, level):
        self.dllHolder.core.world_set_tessellation_level(self.muffInst, level)

    def world_get_tessellation_level(self):
        return self.dllHolder.core.world_get_tessellation_level(self.muffInst)

    def scene_request_retessellation(self):
        return self.dllHolder.scene_request_retessellation(self.muffInst)
        
    def world_reserve_objects_instances(self, meshCount, instanceCount):
        return self.dllHolder.core.world_reserve_objects_instances(self.muffInst, meshCount, instanceCount)
        
    def world_create_object(self, name, flags):
        return self.dllHolder.core.world_create_object(self.muffInst, c_char_p(name.encode('utf-8')), flags)
        
    def world_add_texture(self, path, sampling, mipmaps):
        return self.dllHolder.core.world_add_texture(self.muffInst, c_char_p(path.encode('utf-8')), c_uint32(sampling),
                                                     c_uint32(mipmaps), c_void_p(0), c_void_p(0))
                                                     
    def world_add_texture_value(self, color, channels, sampling):
        return self.dllHolder.core.world_add_texture_value(self.muffInst, color, channels, c_uint32(sampling))
    
    def world_add_material(self, name, matParams):
        return self.dllHolder.core.world_add_material(self.muffInst, c_char_p(name.encode('utf-8')), byref(matParams))
        
    def world_create_instance(self, objHdl, keyframe):
         return self.dllHolder.core.world_create_instance(self.muffInst, objHdl, keyframe)
         
    def world_add_pinhole_camera(self, name, pos, dir, up, count, near, far, fovRad):
        return self.dllHolder.core.world_add_pinhole_camera(self.muffInst, c_char_p(name.encode('utf-8')), byref(pos), byref(dir), byref(up),
                                                            count, near, far, fovRad)
                                                            
    def world_add_focus_camera(self, name, pos, dir, up, count, near, far, focalLength, focusDistance, lensRad, chipHeight):
        return self.dllHolder.core.world_add_focus_camera(self.muffInst, c_char_p(name.encode('utf-8')), byref(pos), byref(dir), byref(up),
                                                          count, near, far, focalLength, focusDistance, lensRad, chipHeight)
                                                            
    def world_add_light(self, name, type, count):
        return self.dllHolder.core.world_add_light(self.muffInst, c_char_p(name.encode('utf-8')), c_uint32(type), count)
    
    def world_set_point_light_position(self, lightHdl, lightPos, index):
        return self.dllHolder.core.world_set_point_light_position(self.muffInst, lightHdl, lightPos, index)
        
    def world_set_point_light_intensity(self, lightHdl, lightIntensity, index):
        return self.dllHolder.core.world_set_point_light_intensity(self.muffInst, lightHdl, lightIntensity, index)
        
    def world_set_spot_light_position(self, lightHdl, pos, index):
        return self.dllHolder.core.world_set_spot_light_position(self.muffInst, lightHdl, pos, index)
        
    def world_set_spot_light_intensity(self, lightHdl, intensity, index):
        return self.dllHolder.core.world_set_spot_light_intensity(self.muffInst, lightHdl, intensity, index)
        
    def world_set_spot_light_direction(self, lightHdl, dir, index):
        return self.dllHolder.core.world_set_spot_light_direction(self.muffInst, lightHdl, dir, index)
        
    def world_set_spot_light_angle(self, lightHdl, angle, index):
        return self.dllHolder.core.world_set_spot_light_angle(self.muffInst, lightHdl, angle, index)
        
    def world_set_spot_light_falloff(self, lightHdl, falloff, index):
        return self.dllHolder.core.world_set_spot_light_falloff(self.muffInst, lightHdl, falloff, index)
        
    def world_set_dir_light_direction(self, lightHdl, direction, index):
        return self.dllHolder.core.world_set_dir_light_direction(self.muffInst, lightHdl, direction, index)
        
    def world_set_dir_light_irradiance(self, lightHdl, irradiance, index):
        return self.dllHolder.core.world_set_dir_light_irradiance(self.muffInst, lightHdl, irradiance, index)
        
    def world_finalize(self, errMsg):
        return self.dllHolder.core.world_finalize(self.muffInst, byref(errMsg))
        
    def world_reserve_scenarios(self, scenarioCount):
        return self.dllHolder.core.world_reserve_scenarios(self.muffInst, scenarioCount)
        
    def world_create_scenario(self, name):
        return self.dllHolder.core.world_create_scenario(self.muffInst, c_char_p(name.encode('utf-8')))
        
    def scenario_set_camera(self, scenarioHdl, camHdl):
        return self.dllHolder.core.scenario_set_camera(self.muffInst, scenarioHdl, camHdl)
        
    def scenario_add_light(self, scenarioHdl, lightHdl):
        return self.dllHolder.core.scenario_add_light(self.muffInst, scenarioHdl, lightHdl)
    
    def world_finalize_scenario(self, scenarioHdl, errMsg):
        return self.dllHolder.core.world_finalize_scenario(self.muffInst, scenarioHdl, byref(errMsg))
        
    def world_load_scenario(self, scenarioHdl):
        return self.dllHolder.core.world_load_scenario(self.muffInst, scenarioHdl)
        
    def mufflon_get_target_image(self, targetName, variance, arrayPtr):
        return self.dllHolder.core.mufflon_get_target_image(self.muffInst, targetName.encode('utf-8'), variance, arrayPtr)
    
    def mufflon_copy_screen_texture_rgba32(self, array, factor):
        return self.dllHolder.core.mufflon_copy_screen_texture_rgba32(self.muffInst, array, c_float(factor))

    def instance_set_transformation_matrix(self, instHdl, mat, isWorldToInst):
        return self.dllHolder.core.instance_set_transformation_matrix(self.muffInst, instHdl, mat, isWorldToInst)


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


class RenderActions:
    screenshotPattern = "#scene-#scenario-#renderer-#iteration-#target"
    sceneName = ""
    activeRendererName = ""

    def __init__(self, binary_path=None, useLoader=True):
        self.dllInterface = DllInterface(binary_path, useLoader)

    def load_json(self, sceneJson, defaultRenderTarget="Radiance"):
        fileName = path_leaf(sceneJson)
        self.sceneName = fileName.split(".")[0]
        returnValue = self.dllInterface.loader_load_json(sceneJson)
        if returnValue != LoaderStatus.SUCCESS:
            raise Exception("Failed to load scene '" + sceneJson + "' (error code: " + returnValue.name + ")")
        self.enable_render_target(defaultRenderTarget, False)

    def enable_renderer(self, rendererName, devices):
        for i in range(self.dllInterface.render_get_renderer_count()):
            name = self.dllInterface.render_get_renderer_name(i)
            shortName = self.dllInterface.render_get_renderer_short_name(i)
            if (name.lower() == rendererName.lower() or shortName.lower() == rendererName.lower()):
                for v in range(self.dllInterface.render_get_renderer_variations(i)):
                    if devices == self.dllInterface.render_get_renderer_devices(i, v):
                        self.activeRendererName = rendererName
                        self.dllInterface.render_enable_renderer(i, v)
                        return
        deviceStr = "[ "
        for dev in Device:
            if (devices & dev) != 0:
                deviceStr += dev.name + " "
        deviceStr += "]"
        raise Exception("Could not find renderer '" + rendererName + "' for the devices " + deviceStr)
        
    def load_scenario(self, scenarioName):
        hdl = self.dllInterface.world_find_scenario(scenarioName)
        if not hdl:
            raise Exception("Failed to find scenario '" + scenarioName + "'")
        if not self.dllInterface.world_load_scenario(hdl):
            raise Exception("Failed to load scenario '" + scenarioName + "'")

    def set_current_animation_frame(self, frame):
        if not self.dllInterface.world_set_frame_current(frame):
            raise Exception("Failed to set animation frame to '" + str(frame) + "'")

    def get_current_animation_frame(self):
        frame = c_uint(0)
        if not self.dllInterface.world_get_frame_current(frame):
            raise Exception("Failed to get current animation frame")
        return frame.value

    def get_animation_frame_count(self):
        frame = c_uint(0)
        if not self.dllInterface.world_get_frame_count(frame):
            raise Exception("Failed to get end animation frame")
        return frame.value
        
    def set_renderer_log_level(self, logLevel):
        if not self.dllInterface.core_set_log_level(logLevel):
            raise Exception("Failed to set log level to '" + logLevel.name + "'")

    def enable_render_target(self, targetName, variance):
        if not self.dllInterface.render_enable_render_target(targetName, variance):
            raise Exception("Failed to enable render target " + targetName + " (variance: " + str(variance) + ")")
        
    def disable_render_target(self, targetName, variance):
      if not self.dllInterface.render_disable_render_target(targetName, variance):
            raise Exception("Failed to disable render target " + targetName + " (variance: " + str(variance) + ")")
            
    def take_denoised_screenshot(self, iterationNr, iterateTime=ProcessTime(0,0), preTime=ProcessTime(0,0), postTime=ProcessTime(0,0)):
        filename = self.screenshotPattern
        filename = filename.replace("#scene", self.sceneName, 1)
        filename = filename.replace("#iterateTime", str(iterateTime.microseconds / 1000) + "ms", 1)
        filename = filename.replace("#preTime", str(iterateTime.microseconds / 1000) + "ms", 1)
        filename = filename.replace("#postTime", str(iterateTime.microseconds / 1000) + "ms", 1)
        filename = filename.replace("#iterateCycles", str(iterateTime.cycles / 1000000) + "MCycles", 1)
        filename = filename.replace("#preCycles", str(preTime.cycles / 1000000) + "MCycles", 1)
        filename = filename.replace("#postCycles", str(postTime.cycles / 1000000) + "MCycles", 1)
        
        self.dllInterface.render_save_denoised_radiance(filename)

    def take_screenshot(self, iterationNr, iterateTime=ProcessTime(0,0), preTime=ProcessTime(0,0), postTime=ProcessTime(0,0)):
        filename = self.screenshotPattern
        filename = filename.replace("#scene", self.sceneName, 1)
        filename = filename.replace("#iterateTime", str(iterateTime.microseconds / 1000) + "ms", 1)
        filename = filename.replace("#preTime", str(iterateTime.microseconds / 1000) + "ms", 1)
        filename = filename.replace("#postTime", str(iterateTime.microseconds / 1000) + "ms", 1)
        filename = filename.replace("#iterateCycles", str(iterateTime.cycles / 1000000) + "MCycles", 1)
        filename = filename.replace("#preCycles", str(preTime.cycles / 1000000) + "MCycles", 1)
        filename = filename.replace("#postCycles", str(postTime.cycles / 1000000) + "MCycles", 1)

        for targetIndex in range(self.dllInterface.render_get_render_target_count()):
            targetName = self.dllInterface.render_get_render_target_name(targetIndex)
            if self.dllInterface.render_is_render_target_enabled(targetName, False):
                self.dllInterface.render_save_screenshot(filename, targetName, False)
            if self.dllInterface.render_is_render_target_enabled(targetName, True):
                self.dllInterface.render_save_screenshot(filename, targetName, True)

    def render_for_iterations(self, iterationCount, printProgress=False, progressSteps=1, takeScreenshot=True, denoise=False):
        accumIterateTime = ProcessTime(0,0)
        accumPreTime = ProcessTime(0,0)
        accumPostTime = ProcessTime(0,0)
        for i in range(iterationCount):
            if printProgress and (i % progressSteps == 0):
                print("--- ", (i + 1), " of ", iterationCount, " ---", flush=True)
            [iterateTime, preTime, postTime] = self.dllInterface.render_iterate()
            accumIterateTime.microseconds += iterateTime.microseconds
            accumIterateTime.cycles += iterateTime.cycles
            accumPreTime.microseconds += preTime.microseconds
            accumPreTime.cycles += preTime.cycles
            accumPostTime.microseconds += postTime.microseconds
            accumPostTime.cycles += postTime.cycles
        if takeScreenshot:
            if denoise:
                self.take_denoised_screenshot(self.dllInterface.render_get_current_iteration(), accumIterateTime, accumPreTime, accumPostTime)
            else:
                self.take_screenshot(self.dllInterface.render_get_current_iteration(), accumIterateTime, accumPreTime, accumPostTime)

    def render_for_seconds(self, secondsToRender):
        startTime = process_time()
        curTime = startTime
        while curTime - startTime < secondsToRender:
            self.dllInterface.render_iterate()
            curTime = process_time()
        self.take_screenshot(self.dllInterface.render_get_current_iteration())

    def render_reset(self):
        self.dllInterface.render_reset()

    def renderer_set_parameter_bool(self, parameterName, value):
        return self.dllInterface.renderer_set_parameter_bool(parameterName, value)

    def renderer_set_parameter_float(self, parameterName, value):
        return self.dllInterface.renderer_set_parameter_float(parameterName, value)

    def renderer_set_parameter_int(self, parameterName, value):
        return self.dllInterface.renderer_set_parameter_int(parameterName, value)

    def renderer_set_parameter_enum(self, parameterName, valueName):
        return self.dllInterface.renderer_set_parameter_enum(parameterName, self.dllInterface.renderer_get_parameter_enum_value(parameterName, valueName))
