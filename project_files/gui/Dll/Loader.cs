using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace gui.Dll
{

    /// <summary>
    /// DLL communication with mffloader.dll
    /// </summary>
    public static class Loader
    {
        public enum LoaderStatus
        {
            SUCCESS,
            ERROR,
            ABORT
        };

        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool loader_set_logger(Core.LogCallback callback);
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern LoaderStatus loader_load_json(string path);
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool loader_abort();
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void loader_profiling_enable();
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void loader_profiling_disable();
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Boolean loader_profiling_set_level(Core.ProfilingLevel level);
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Boolean loader_profiling_save_current_state(string path);
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Boolean loader_profiling_save_snapshots(string path);
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Boolean loader_profiling_save_total_and_snapshots(string path);
        [DllImport("mffloader.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern string loader_profiling_get_current_state();
        [DllImport("mffloader.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern string loader_profiling_get_snapshots();
        [DllImport("mffloader.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern string loader_profiling_get_total_and_snapshots();
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void loader_profiling_reset();
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string loader_get_dll_error();
        [DllImport("mffloader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool loader_set_log_level(Core.Severity level);
    }
}
