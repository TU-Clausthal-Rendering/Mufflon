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


        public delegate void LogCallback(string message, Severity severity);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool display_screenshot();

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool resize(int width, int height, int offsetX, int offsetY);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string core_get_dll_error();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool core_set_log_level(Severity level);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void execute_command(string command);

        // World API
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_create_object();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_create_instance(IntPtr obj);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_create_scenario(string name);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_find_scenario(string name);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_add_lambert_material(string name, Vec3 rgb);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_add_lambert_material_textured(string name, IntPtr texture);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_add_pinhole_camera(string name, Vec3 position,
            Vec3 dir, Vec3 up, float near, float far, float vFov);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_add_point_light(string name, Vec3 position,
										   Vec3 intensity);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_add_spot_light(string name, Vec3 position,
            Vec3 direction, Vec3 intensity, float openingAngleRad,
            float falloffStartRad);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_add_directional_light(string name, Vec3 direction,
            Vec3 radiance);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_add_envmap_light(string name, IntPtr envmap);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ulong world_get_camera_count();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_get_camera(string name);
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
        internal static extern LightType world_get_light_type(string name);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string world_get_point_light_by_index(ulong index, ref IntPtr hdl);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string world_get_spot_light_by_index(ulong index, ref IntPtr hdl);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string world_get_dir_light_by_index(ulong index, ref IntPtr hdl);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string world_get_env_light_by_index(ulong index, ref IntPtr hdl);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_get_light(string name, LightType type);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_load_scenario(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_get_current_scene();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_get_current_scenario();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint world_get_scenario_count();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string world_get_scenario_name_by_index(uint index);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string world_get_scenario_name(IntPtr hdl);

        // Scenario API
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string scenario_get_name(IntPtr scenario);
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
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string scenario_get_light_name(IntPtr scenario, ulong index);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_add_light(IntPtr scenario, string name);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_remove_light_by_index(IntPtr scenario, ulong index);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool scenario_remove_light_by_named(IntPtr scenario, string name);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ushort scenario_declare_material_slot(IntPtr scenario, string name);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern ushort scenario_get_material_slot(IntPtr scenario, string name);
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
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string world_get_env_light_map(IntPtr hdl);

        // Camera API
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern CameraType world_get_camera_type(IntPtr cam);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string world_get_camera_name(IntPtr cam);
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

        // Renderer API
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_enable_renderer(RendererType type);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_iterate();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_reset();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_get_screenshot();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool render_save_screenshot(string filename);
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
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string renderer_get_parameter_desc(uint idx, ref ParameterType type);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool renderer_set_parameter_int(string name, int value);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool renderer_get_parameter_int(string name, ref int value);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool renderer_set_parameter_float(string name, float value);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool renderer_get_parameter_float(string name, ref float value);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool renderer_set_parameter_bool(string name, uint value);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool renderer_get_parameter_bool(string name, ref uint value);

        // Interface for profiling
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void profiling_enable();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void profiling_disable();
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool profiling_set_level(ProfilingLevel level);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool profiling_save_current_state(string path);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool profiling_save_snapshots(string path);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool profiling_save_total_and_snapshots(string path);
        [DllImport("core.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string profiling_get_current_state();
        [DllImport("core.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string profiling_get_snapshots();
        [DllImport("core.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string profiling_get_total_and_snapshots();
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
    }
}
