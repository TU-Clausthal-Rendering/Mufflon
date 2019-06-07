using gui.Utility;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using gui.Model.Light;

namespace gui.Dll
{
    /// <summary>
    /// DLL communication with core.dll
    /// </summary>
    static public class Core
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct Vec3
        {
            public float x, y, z;
            public Vec3(float a, float b, float c)
            {
                x = a;
                y = b;
                z = c;
            }
            public Vec3(Vec3<float> vec)
            {
                x = vec.X;
                y = vec.Y;
                z = vec.Z;
            }

            public Vec3<float> ToUtilityVec()
            {
                return new Utility.Vec3<float>(x, y, z);
            }
        };

        [StructLayout(LayoutKind.Sequential)]
        public struct ProcessTime
        {
            public UInt64 cycles;
            public UInt64 microseconds;

            public static ProcessTime operator+(ProcessTime t1, ProcessTime t2)
            {
                t1.cycles += t2.cycles;
                t1.microseconds += t2.microseconds;
                return t1;
            }
        };

        internal enum MaterialType
        {
            Lambert
        };

        public enum CameraType
        {
            Pinhole,
            Focus
        };

        public enum LightType
        {
            Point,
            Spot,
            Directional,
            Envmap
        };

        public static LightType FromModelLightType(LightModel.LightType type)
        {
            switch (type)
            {
                case LightModel.LightType.Point:
                    return LightType.Point;
                case LightModel.LightType.Directional:
                    return LightType.Directional;
                case LightModel.LightType.Spot:
                    return LightType.Spot;
                case LightModel.LightType.Envmap:
                    return LightType.Envmap;
                case LightModel.LightType.Goniometric:
                    default:
                    throw new NotImplementedException();
            }
        }

        public enum ProfilingLevel
        {
            All,
            High,
            Low,
            Off
        };

        public enum ParameterType
        {
            Int,
            Float,
            Bool
        };

        public enum Severity
        {
            Pedantic,
            Info,
            Warning,
            Error,
            FatalError
        };

        public enum TextureSampling
        {
            Nearest,
            Linear
        };

        [Flags]
        public enum RenderDevice
        {
            None = 0,
            Cpu = 1,
            Cuda = 2,
            OpenGL = 4
        };

        public enum TextureFormat
        {
            R8U,
            RG8U,
            RGBA8U,
            R16U,
            RG16U,
            RGBA16U,
            R16F,
            RG16F,
            RGBA16F,
            R32F,
            RG32F,
            RGBA32F,
            Invalid
        };

        public delegate void LogCallback(string message, Severity severity);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool core_get_target_image(UInt32 index, Boolean variance, out IntPtr ptr);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool core_get_target_image_num_channels(IntPtr numChannels);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool core_copy_screen_texture_rgba32(IntPtr ptr, float factor);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool core_get_pixel_info(uint x, uint y, Boolean borderClamp, out float r,
            out float g, out float b, out float a);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "core_get_dll_error")]
        private static extern IntPtr core_get_dll_error_();
        internal static string core_get_dll_error() { return StringUtil.FromNativeUTF8(core_get_dll_error_()); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool core_set_log_level(Severity level);

        // World API
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void world_clear_all();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_create_object();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_object_name")]
        private static extern IntPtr world_get_object_name_(IntPtr obj);
        internal static string world_get_object_name(IntPtr obj) { return StringUtil.FromNativeUTF8(world_get_object_name_(obj)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_create_instance(IntPtr obj);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_create_scenario")]
        private static extern IntPtr world_create_scenario_(IntPtr name);
        internal static IntPtr world_create_scenario(string name) { return world_create_scenario_(StringUtil.ToNativeUtf8(name)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_find_scenario")]
        private static extern IntPtr world_find_scenario_(IntPtr name);
        internal static IntPtr world_find_scenario(string name) { return world_find_scenario_(StringUtil.ToNativeUtf8(name)); }
        // TODO: material interface
        //internal static IntPtr world_add_material(string name, ...);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_add_pinhole_camera")]
        private static extern IntPtr world_add_pinhole_camera_(IntPtr name, Vec3 position,
            Vec3 dir, Vec3 up, float near, float far, float vFov);
        internal static IntPtr world_add_pinhole_camera(string name, Vec3 position,
            Vec3 dir, Vec3 up, float near, float far, float vFov) { return world_add_pinhole_camera_(StringUtil.ToNativeUtf8(name), position, dir, up, near, far, vFov); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_add_focus_camera")]
        private static extern IntPtr world_add_focus_camera_(IntPtr name, Vec3 position, Vec3 dir,
            Vec3 up, float near, float far, float focalLength, float focusDistance,
            float lensRad, float chipHeight);
        internal static IntPtr world_add_focus_camera(string name, Vec3 position, Vec3 dir,
            Vec3 up, float near, float far, float focalLength, float focusDistance,
            float lensRad, float chipHeight) { return world_add_focus_camera_(StringUtil.ToNativeUtf8(name), position, dir, up, near, far, focalLength, focusDistance, lensRad, chipHeight); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_remove_camera(IntPtr hdl);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_add_light")]
        private static extern IntPtr world_add_light_(IntPtr name, LightType type);
        internal static IntPtr world_add_light(string name, LightType type) { return world_add_light_(StringUtil.ToNativeUtf8(name), type); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_remove_light(IntPtr hdl);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ulong world_get_camera_count();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera")]
        private static extern IntPtr world_get_camera_(IntPtr name);
        internal static IntPtr world_get_camera(string name) { return world_get_camera_(StringUtil.ToNativeUtf8(name)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_get_camera_by_index(ulong index);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ulong world_get_point_light_count();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ulong world_get_spot_light_count();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ulong world_get_dir_light_count();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ulong world_get_env_light_count();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_get_light_handle(ulong index, LightType type);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_light_name")]
        private static extern IntPtr world_get_light_name_(IntPtr hdl);
        internal static string world_get_light_name(IntPtr hdl) { return StringUtil.FromNativeUTF8(world_get_light_name_(hdl)); }

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_add_texture")]
        private static extern IntPtr world_add_texture_(IntPtr path, TextureSampling sampling);
        internal static IntPtr world_add_texture(string path, TextureSampling sampling) { return world_add_texture_(StringUtil.ToNativeUtf8(path), sampling); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_add_texture_value(out float[] value, int num, TextureSampling sampling);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_load_scenario(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_get_current_scene();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_get_current_scenario();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint world_get_scenario_count();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_get_scenario_by_index(uint index);
        
        // Scenario API
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_name")]
        private static extern IntPtr scenario_get_name_(IntPtr scenario);
        internal static string scenario_get_name(IntPtr scenario) { return StringUtil.FromNativeUTF8(scenario_get_name_(scenario)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ulong scenario_get_global_lod_level(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_set_global_lod_level(IntPtr scenario, ulong level);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_get_resolution(IntPtr scenario, out uint width, out uint height);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_set_resolution(IntPtr scenario, uint width, uint height);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr scenario_get_camera(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_frame_current(uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_frame_current(out uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_frame_start(out uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_frame_end(out uint frame);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_set_camera(IntPtr scenario, IntPtr cam);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_is_object_masked(IntPtr scenario, IntPtr obj);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_mask_object(IntPtr scenario, IntPtr obj);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ulong scenario_get_object_lod(IntPtr scenario, IntPtr obj);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_set_object_lod(IntPtr scenario, IntPtr obj,
            ulong level);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Int32 scenario_get_point_light_count(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Int32 scenario_get_spot_light_count(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Int32 scenario_get_dir_light_count(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_has_envmap_light(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr scenario_get_light_handle(IntPtr scenario, ulong index, LightType type);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_add_light(IntPtr scenario, IntPtr hdl);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_remove_light(IntPtr scenario, IntPtr hdl);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_declare_material_slot")]
        private static extern ushort scenario_declare_material_slot_(IntPtr scenario, IntPtr name);
        internal static ushort scenario_declare_material_slot(IntPtr scenario, string name) { return scenario_declare_material_slot_(scenario, StringUtil.ToNativeUtf8(name)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_material_slot")]
        private static extern ushort scenario_get_material_slot_(IntPtr scenario, IntPtr name);
        internal static ushort scenario_get_material_slot(IntPtr scenario, string name) { return scenario_get_material_slot_(scenario, StringUtil.ToNativeUtf8(name)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr scenario_get_assigned_material(IntPtr scenario, ushort index);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_assign_material(IntPtr scenario, ushort index,
            IntPtr handle);

        // Light API
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_point_light_path_segments(IntPtr hdl, out uint count);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_point_light_position(IntPtr hdl, out Vec3 pos, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_point_light_intensity(IntPtr hdl, out Vec3 intensity, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_point_light_position(IntPtr hdl, Vec3 pos, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_point_light_intensity(IntPtr hdl, Vec3 intensity, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_spot_light_path_segments(IntPtr hdl, out uint count);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_spot_light_position(IntPtr hdl, out Vec3 pos, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_spot_light_intensity(IntPtr hdl, out Vec3 intensity, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_spot_light_direction(IntPtr hdl, out Vec3 direction, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_spot_light_angle(IntPtr hdl, out float angle, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_spot_light_falloff(IntPtr hdl, out float falloff, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_spot_light_position(IntPtr hdl, Vec3 pos, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_spot_light_intensity(IntPtr hdl, Vec3 intensity, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_spot_light_direction(IntPtr hdl, Vec3 direction, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_spot_light_angle(IntPtr hdl, float angle, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_spot_light_falloff(IntPtr hdl, float fallof, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_dir_light_path_segments(IntPtr hdl, out uint count);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_dir_light_direction(IntPtr hdl, out Vec3 direction, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_dir_light_irradiance(IntPtr hdl, out Vec3 irradiance, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_dir_light_direction(IntPtr hdl, Vec3 direction, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_dir_light_irradiance(IntPtr hdl, Vec3 irradiance, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_env_light_map")]
        private static extern IntPtr world_get_env_light_map_(IntPtr hdl);
        internal static string world_get_env_light_map(IntPtr hdl) { return StringUtil.FromNativeUTF8(world_get_env_light_map_(hdl)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_env_light_map(IntPtr hdl, IntPtr tex);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void world_set_tessellation_level(float level);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern float world_get_tessellation_level();

        // Camera API
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern CameraType world_get_camera_type(IntPtr cam);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_name")]
        private static extern IntPtr world_get_camera_name_(IntPtr cam);
        internal static string world_get_camera_name(IntPtr cam) { return StringUtil.FromNativeUTF8(world_get_camera_name_(cam)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_camera_path_segment_count(IntPtr cam, out uint segments);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_camera_position(IntPtr cam, out Vec3 pos, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_camera_current_position(IntPtr cam, out Vec3 pos);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_camera_direction(IntPtr cam, out Vec3 dir, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_camera_current_direction(IntPtr cam, out Vec3 dir);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_camera_up(IntPtr cam, out Vec3 up, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_camera_current_up(IntPtr cam, out Vec3 up);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_camera_near(IntPtr cam, out float near);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_camera_far(IntPtr cam, out float far);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_camera_position(IntPtr cam, Vec3 pos, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_camera_current_position(IntPtr cam, Vec3 pos);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_camera_direction(IntPtr cam, Vec3 dir, Vec3 up, uint frame);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_camera_current_direction(IntPtr cam, Vec3 dir, Vec3 up);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_camera_near(IntPtr cam, float near);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_camera_far(IntPtr cam, float far);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_pinhole_camera_fov(IntPtr cam, out float vFov);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_pinhole_camera_fov(IntPtr cam, float vFov);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_focus_camera_focal_length(IntPtr cam, out float focalLength);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_focus_camera_focus_distance(IntPtr cam, out float focusDistance);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_focus_camera_sensor_height(IntPtr cam, out float sensorHeight);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_focus_camera_aperture(IntPtr cam, out float aperture);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_focus_camera_focal_length(IntPtr cam, float focalLength);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_focus_camera_focus_distance(IntPtr cam, float focusDistance);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_focus_camera_sensor_height(IntPtr cam, float sensorHeight);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_focus_camera_aperture(IntPtr cam, float aperture);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scene_move_active_camera(float x, float y, float z);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scene_rotate_active_camera(float x, float y, float z);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scene_is_sane();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scene_get_bounding_box(IntPtr scene, out Vec3 min, out Vec3 max);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scene_request_retessellation();

        // Renderer API
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern UInt32 render_get_renderer_count();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern UInt32 render_get_renderer_variations(uint index);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_get_renderer_name")]
        private static extern IntPtr render_get_renderer_name_(UInt32 index);
        internal static string render_get_renderer_name(UInt32 index) { return StringUtil.FromNativeUTF8(render_get_renderer_name_(index)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_get_renderer_short_name")]
        private static extern IntPtr render_get_renderer_short_name_(UInt32 index);
        internal static string render_get_renderer_short_name(UInt32 index) { return StringUtil.FromNativeUTF8(render_get_renderer_short_name_(index)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern RenderDevice render_get_renderer_devices(UInt32 index, uint variations);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_enable_renderer(UInt32 index, uint variation);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_iterate(out ProcessTime iterateTime, out ProcessTime preTime, out ProcessTime postTime);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_reset();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern UInt32 render_get_current_iteration();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_save_screenshot")]
        private static extern bool render_save_screenshot_(IntPtr filename, UInt32 targetIndex, UInt32 variance);
        internal static bool render_save_screenshot(string filename, UInt32 targetIndex, UInt32 variance) { return render_save_screenshot_(StringUtil.ToNativeUtf8(filename), targetIndex, variance); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern UInt32 render_get_render_target_count();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_get_render_target_name")]
        internal static extern IntPtr render_get_render_target_name_(UInt32 index);
        internal static string render_get_render_target_name(UInt32 index) { return StringUtil.FromNativeUTF8(render_get_render_target_name_(index)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_enable_render_target(UInt32 index, uint variance);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_disable_render_target(UInt32 inddex, uint variance);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_enable_non_variance_render_targets();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_enable_all_render_targets();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_disable_variance_render_targets();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_disable_all_render_targets();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_is_render_target_enabled(UInt32 index, Boolean variance);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint renderer_get_num_parameters();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_desc")]
        private static extern IntPtr renderer_get_parameter_desc_(uint idx, out ParameterType type);
        internal static string renderer_get_parameter_desc(uint idx, out ParameterType type) { return StringUtil.FromNativeUTF8(renderer_get_parameter_desc_(idx, out type)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_set_parameter_int")]
        private static extern bool renderer_set_parameter_int_(IntPtr name, int value);
        internal static bool renderer_set_parameter_int(string name, int value) { return renderer_set_parameter_int_(StringUtil.ToNativeUtf8(name), value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_int")]
        private static extern bool renderer_get_parameter_int_(IntPtr name, out int value);
        internal static bool renderer_get_parameter_int(string name, out int value) { return renderer_get_parameter_int_(StringUtil.ToNativeUtf8(name), out value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_set_parameter_float")]
        private static extern bool renderer_set_parameter_float_(IntPtr name, float value);
        internal static bool renderer_set_parameter_float(string name, float value) { return renderer_set_parameter_float_(StringUtil.ToNativeUtf8(name), value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_float")]
        private static extern bool renderer_get_parameter_float_(IntPtr name, out float value);
        internal static bool renderer_get_parameter_float(string name, out float value) { return renderer_get_parameter_float_(StringUtil.ToNativeUtf8(name), out value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_set_parameter_bool")]
        private static extern bool renderer_set_parameter_bool_(IntPtr name, uint value);
        internal static bool renderer_set_parameter_bool(string name, uint value) { return renderer_set_parameter_bool_(StringUtil.ToNativeUtf8(name), value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_bool")]
        private static extern bool renderer_get_parameter_bool_(IntPtr name, out uint value);
        internal static bool renderer_get_parameter_bool(string name, out uint value) { return renderer_get_parameter_bool_(StringUtil.ToNativeUtf8(name), out value); }

        // Interface for profiling
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void profiling_enable();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void profiling_disable();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool profiling_set_level(ProfilingLevel level);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_save_current_state")]
        private static extern bool profiling_save_current_state_(IntPtr path);
        internal static bool profiling_save_current_state(string path) { return profiling_save_current_state_(StringUtil.ToNativeUtf8(path)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_save_snapshots")]
        private static extern bool profiling_save_snapshots_(IntPtr path);
        internal static bool profiling_save_snapshots(string path) { return profiling_save_snapshots_(StringUtil.ToNativeUtf8(path)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_save_total_and_snapshots")]
        private static extern bool profiling_save_total_and_snapshots_(IntPtr path);
        internal static bool profiling_save_total_and_snapshots(string path) { return profiling_save_total_and_snapshots_(StringUtil.ToNativeUtf8(path)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_get_current_state")]
        private static extern IntPtr profiling_get_current_state_();
        internal static string profiling_get_current_state() { return StringUtil.FromNativeUTF8(profiling_get_current_state_()); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_get_snapshots")]
        private static extern IntPtr profiling_get_snapshots_();
        internal static string profiling_get_snapshots() { return StringUtil.FromNativeUTF8(profiling_get_snapshots_()); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_get_total")]
        private static extern IntPtr profiling_get_total_();
        internal static string profiling_get_total() { return StringUtil.FromNativeUTF8(profiling_get_total_()); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "profiling_get_total_and_snapshots")]
        private static extern IntPtr profiling_get_total_and_snapshots_();
        internal static string profiling_get_total_and_snapshots() { return StringUtil.FromNativeUTF8(profiling_get_total_and_snapshots_()); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void profiling_reset();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ulong profiling_get_total_cpu_memory();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ulong profiling_get_free_cpu_memory();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ulong profiling_get_used_cpu_memory();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ulong profiling_get_total_gpu_memory();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ulong profiling_get_free_gpu_memory();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ulong profiling_get_used_gpu_memory();

        // Interface for initialization and destruction
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool mufflon_initialize();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool mufflon_initialize_opengl();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool mufflon_set_logger(LogCallback logCallback);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int mufflon_get_cuda_device_index();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool mufflon_is_cuda_available();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void mufflon_destroy();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void mufflon_destroy_opengl();

        //[DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "get_teststring")]
        //private static extern IntPtr get_teststring_();
        //internal static string get_teststring() { return StringUtil.FromNativeUTF8(get_teststring_()); }
    }
}
