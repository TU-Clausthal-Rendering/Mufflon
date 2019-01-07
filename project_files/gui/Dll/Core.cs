using gui.Utility;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

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
        };

        internal enum MaterialType
        {
            LAMBERT
        };

        public enum CameraType
        {
            PINHOLE,
            FOCUS
        };

        public enum RendererType
        {
            CPU_PT,
            GPU_PT
        };

        internal enum RenderTarget
        {
            RADIANCE,
            POSITION,
            ALBEDO,
            NORMAL,
            LIGHTNESS
        };

        public enum LightType
        {
            POINT,
            SPOT,
            DIRECTIONAL,
            ENVMAP
        };

        public enum ProfilingLevel
        {
            ALL,
            HIGH,
            LOW,
            OFF
        };

        public enum ParameterType
        {
            PARAM_INT,
            PARAM_FLOAT,
            PARAM_BOOL
        };

        public enum Severity
        {
            PEDANTIC,
            INFO,
            WARNING,
            ERROR,
            FATAL_ERROR
        };

        public enum TextureSampling
        {
            NEAREST,
            LINEAR
        };

        public delegate void LogCallback(string message, Severity severity);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool display_screenshot();

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool resize(int width, int height, int offsetX, int offsetY);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "core_get_dll_error")]
        private static extern IntPtr core_get_dll_error_();
        internal static string core_get_dll_error() { return StringUtil.FromNativeUTF8(core_get_dll_error_()); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool core_set_log_level(Severity level);

        // World API
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
        internal static extern IntPtr world_reload_current_scenario();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_get_current_scene();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_get_current_scenario();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint world_get_scenario_count();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_scenario_name_by_index")]
        private static extern IntPtr world_get_scenario_name_by_index_(uint index);
        internal static string world_get_scenario_name_by_index(uint index) { return StringUtil.FromNativeUTF8(world_get_scenario_name_by_index_(index)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_scenario_name")]
        private static extern IntPtr world_get_scenario_name_(IntPtr hdl);
        internal static string world_get_scenario_name(IntPtr hdl) { return StringUtil.FromNativeUTF8(world_get_scenario_name_(hdl)); }

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
        internal static extern uint scenario_get_light_count(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_get_light_name")]
        private static extern IntPtr scenario_get_light_name_(IntPtr scenario, ulong index);
        internal static string scenario_get_light_name(IntPtr scenario, ulong index) { return StringUtil.FromNativeUTF8(scenario_get_light_name_(scenario, index)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_add_light")]
        private static extern bool scenario_add_light_(IntPtr scenario, IntPtr name);
        internal static bool scenario_add_light(IntPtr scenario, string name) { return scenario_add_light_(scenario, StringUtil.ToNativeUtf8(name)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_remove_light_by_index(IntPtr scenario, ulong index);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "scenario_remove_light_by_named")]
        private static extern bool scenario_remove_light_by_named_(IntPtr scenario, IntPtr name);
        internal static bool scenario_remove_light_by_named(IntPtr scenario, string name) { return scenario_remove_light_by_named_(scenario, StringUtil.ToNativeUtf8(name)); }
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
        internal static extern bool world_get_point_light_position(IntPtr hdl, ref Vec3 pos);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_point_light_intensity(IntPtr hdl, ref Vec3 intensity);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_point_light_position(IntPtr hdl, Vec3 pos);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_point_light_intensity(IntPtr hdl, Vec3 intensity);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_spot_light_position(IntPtr hdl, ref Vec3 pos);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_spot_light_intensity(IntPtr hdl, ref Vec3 intensity);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_spot_light_direction(IntPtr hdl, ref Vec3 direction);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_spot_light_angle(IntPtr hdl, ref float angle);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_spot_light_falloff(IntPtr hdl, ref float falloff);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_spot_light_position(IntPtr hdl, Vec3 pos);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_spot_light_intensity(IntPtr hdl, Vec3 intensity);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_spot_light_direction(IntPtr hdl, Vec3 direction);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_spot_light_angle(IntPtr hdl, float angle);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_spot_light_falloff(IntPtr hdl, float fallof);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_dir_light_direction(IntPtr hdl, ref Vec3 direction);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_dir_light_radiance(IntPtr hdl, ref Vec3 radiance);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_dir_light_direction(IntPtr hdl, Vec3 direction);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_dir_light_radiance(IntPtr hdl, Vec3 radiance);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_env_light_map")]
        private static extern IntPtr world_get_env_light_map_(IntPtr hdl);
        internal static string world_get_env_light_map(IntPtr hdl) { return StringUtil.FromNativeUTF8(world_get_env_light_map_(hdl)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_env_light_map(IntPtr hdl, IntPtr tex);

        // Camera API
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern CameraType world_get_camera_type(IntPtr cam);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "world_get_camera_name")]
        private static extern IntPtr world_get_camera_name_(IntPtr cam);
        internal static string world_get_camera_name(IntPtr cam) { return StringUtil.FromNativeUTF8(world_get_camera_name_(cam)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_camera_position(IntPtr cam, ref Vec3 pos);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_camera_direction(IntPtr cam, ref Vec3 dir);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_camera_up(IntPtr cam, ref Vec3 up);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_camera_near(IntPtr cam, ref float near);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_camera_far(IntPtr cam, ref float far);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_camera_position(IntPtr cam, Vec3 pos);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_camera_direction(IntPtr cam, Vec3 dir);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_camera_up(IntPtr cam, Vec3 up);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_camera_near(IntPtr cam, float near);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_camera_far(IntPtr cam, float far);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_pinhole_camera_fov(IntPtr cam, ref float vFov);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_pinhole_camera_fov(IntPtr cam, float vFov);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_focus_camera_focal_length(IntPtr cam, ref float focalLength);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_focus_camera_focus_distance(IntPtr cam, ref float focusDistance);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_focus_camera_sensor_height(IntPtr cam, ref float sensorHeight);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_get_focus_camera_aperture(IntPtr cam, ref float aperture);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_focus_camera_focal_length(IntPtr cam, float focalLength);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_focus_camera_focus_distance(IntPtr cam, float focusDistance);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_focus_camera_sensor_height(IntPtr cam, float sensorHeight);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool world_set_focus_camera_aperture(IntPtr cam, float aperture);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void scene_mark_lighttree_dirty();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void scene_mark_envmap_dirty();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scene_move_active_camera(float x, float y, float z);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scene_rotate_active_camera(float x, float y, float z);

        // Renderer API
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_enable_renderer(RendererType type);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_iterate();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_reset();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_get_screenshot();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "render_save_screenshot")]
        private static extern bool render_save_screenshot_(IntPtr filename);
        internal static bool render_save_screenshot(string filename) { return render_save_screenshot_(StringUtil.ToNativeUtf8(filename)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_enable_render_target(RenderTarget target, uint variance);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_disable_render_target(RenderTarget target, uint variance);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_enable_variance_render_targets();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_enable_non_variance_render_targets();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_enable_all_render_targets();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_disable_variance_render_targets();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_disable_non_variance_render_targets();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_disable_all_render_targets();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint renderer_get_num_parameters();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_desc")]
        private static extern IntPtr renderer_get_parameter_desc_(uint idx, ref ParameterType type);
        internal static string renderer_get_parameter_desc(uint idx, ref ParameterType type) { return StringUtil.FromNativeUTF8(renderer_get_parameter_desc_(idx, ref type)); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_set_parameter_int")]
        private static extern bool renderer_set_parameter_int_(IntPtr name, int value);
        internal static bool renderer_set_parameter_int(string name, int value) { return renderer_set_parameter_int_(StringUtil.ToNativeUtf8(name), value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_int")]
        private static extern bool renderer_get_parameter_int_(IntPtr name, ref int value);
        internal static bool renderer_get_parameter_int(string name, ref int value) { return renderer_get_parameter_int_(StringUtil.ToNativeUtf8(name), ref value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_set_parameter_float")]
        private static extern bool renderer_set_parameter_float_(IntPtr name, float value);
        internal static bool renderer_set_parameter_float(string name, float value) { return renderer_set_parameter_float_(StringUtil.ToNativeUtf8(name), value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_float")]
        private static extern bool renderer_get_parameter_float_(IntPtr name, ref float value);
        internal static bool renderer_get_parameter_float(string name, ref float value) { return renderer_get_parameter_float_(StringUtil.ToNativeUtf8(name), ref value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_set_parameter_bool")]
        private static extern bool renderer_set_parameter_bool_(IntPtr name, uint value);
        internal static bool renderer_set_parameter_bool(string name, uint value) { return renderer_set_parameter_bool_(StringUtil.ToNativeUtf8(name), value); }
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "renderer_get_parameter_bool")]
        private static extern bool renderer_get_parameter_bool_(IntPtr name, ref uint value);
        internal static bool renderer_get_parameter_bool(string name, ref uint value) { return renderer_get_parameter_bool_(StringUtil.ToNativeUtf8(name), ref value); }

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
        internal static extern bool mufflon_initialize(LogCallback logCallback);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int mufflon_get_cuda_device_index();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool mufflon_is_cuda_available();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void mufflon_destroy();

        //[DllImport("core.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "get_teststring")]
        //private static extern IntPtr get_teststring_();
        //internal static string get_teststring() { return StringUtil.FromNativeUTF8(get_teststring_()); }
    }
}
