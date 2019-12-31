using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using gui.Utility;

namespace gui.Dll
{

    /// <summary>
    /// DLL communication with mffloader.dll
    /// </summary>
    public static class Loader
    {
        private static IntPtr mffLoaderInstHdl;

        public enum LoaderStatus
        {
            SUCCESS,
            ERROR,
            ABORT
        };

        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_get_dll_error")]
        private static extern IntPtr loader_get_dll_error_();
        internal static string loader_get_dll_error() { return StringUtil.FromNativeUTF8(loader_get_dll_error_()); }
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_initialize")]
        private static extern IntPtr loader_initialize_(IntPtr mffInstHdl);
        internal static bool loader_initialize() { mffLoaderInstHdl = loader_initialize_(Core.muffInstHdl); return mffLoaderInstHdl != IntPtr.Zero; }
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_destroy")]
        private static extern void loader_destroy_(IntPtr instHdl);
        internal static void loader_destroy() { loader_destroy_(mffLoaderInstHdl); }
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_load_json")]
        private static extern LoaderStatus loader_load_json_(IntPtr instHdl, IntPtr path);
        internal static LoaderStatus loader_load_json(string path) { return loader_load_json_(mffLoaderInstHdl, StringUtil.ToNativeUtf8(path)); }
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_save_scene")]
        private static extern LoaderStatus loader_save_scene_(IntPtr instHdl, IntPtr path);
        internal static LoaderStatus loader_save_scene(string path) { return loader_save_scene_(mffLoaderInstHdl, StringUtil.ToNativeUtf8(path)); }
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_load_lod")]
        private static extern Boolean loader_load_lod_(IntPtr instHdl, IntPtr obj, UInt32 lod);
        internal static Boolean loader_load_lod(IntPtr obj, UInt32 lod) { return loader_load_lod_(mffLoaderInstHdl, obj, lod); }
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_abort")]
        private static extern Boolean loader_abort_(IntPtr instHdl);
        internal static Boolean loader_abort() { return loader_abort_(mffLoaderInstHdl); }
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_get_loading_status")]
        private static extern IntPtr loader_get_loading_status_(IntPtr instHdl);
        internal static string loader_get_loading_status() { return StringUtil.FromNativeUTF8(loader_get_loading_status_(mffLoaderInstHdl)); }
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_profiling_enable")]
        internal static extern void loader_profiling_enable();
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_profiling_disable")]
        internal static extern void loader_profiling_disable();
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_profiling_set_level")]
        internal static extern Boolean loader_profiling_set_level(Core.ProfilingLevel level);
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_profiling_save_current_state")]
        internal static extern Boolean loader_profiling_save_total_and_snapshots(string path);
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_profiling_get_current_state")]
        private static extern IntPtr loader_profiling_get_current_state_();
        internal static string loader_profiling_get_current_state() { return StringUtil.FromNativeUTF8(loader_profiling_get_current_state_()); }
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_profiling_get_snapshots")]
        private static extern IntPtr loader_profiling_get_snapshots_();
        internal static string loader_profiling_get_snapshots() { return StringUtil.FromNativeUTF8(loader_profiling_get_snapshots_()); }
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_profiling_get_total")]
        private static extern IntPtr loader_profiling_get_total_();
        internal static string loader_profiling_get_total() { return StringUtil.FromNativeUTF8(loader_profiling_get_total_()); }
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_profiling_get_total_and_snapshots")]
        private static extern IntPtr loader_profiling_get_total_and_snapshots_();
        internal static string loader_profiling_get_total_and_snapshots() { return StringUtil.FromNativeUTF8(loader_profiling_get_total_and_snapshots_()); }
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "loader_profiling_reset")]
        internal static extern void loader_profiling_reset();
    }
}
