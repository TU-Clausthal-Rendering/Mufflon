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
    static class Core
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct Vec3
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

        internal enum CameraType
        {
            PINHOLE
        };

        internal enum RendererType
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

        internal enum LightType
        {
            POINT,
            SPOT,
            DIRECTIONAL,
            ENVMAP
        };

        internal enum ProfilingLevel
        {
            OFF,
            LOW,
            HIGH,
            ALL
        };


        public delegate void LogCallback(string message, int severity);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool display_screenshot();

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool resize(int width, int height, int offsetX, int offsetY);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr get_error(out int length);

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
        internal static extern IntPtr world_get_camera(string name);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_get_light(string name, LightType type);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_load_scenario(IntPtr scenario);
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr world_get_current_scene();

        // Scenario API
        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
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
        internal static extern void mufflon_destroy();

        /// <summary>
        /// wrapper for get_error()
        /// </summary>
        /// <returns>error string returned by get_error()</returns>
        internal static string GetDllError()
        {
            var ptr = get_error(out var length);
            return ptr.Equals(IntPtr.Zero) ? "" : Marshal.PtrToStringAnsi(ptr, length);
        }
    }
}
