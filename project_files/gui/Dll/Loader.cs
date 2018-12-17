using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace gui.Dll
{
    /// <summary>
    /// DLL communication with loader.dll
    /// </summary>
    static class Loader
    {
        internal enum ProfilingLevel
        {
            OFF,
            LOW,
            HIGH,
            ALL
        };

        [DllImport("loader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool loader_set_logger(Core.LogCallback callback);
        [DllImport("loader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool loader_load_json(string path);
        [DllImport("loader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void loader_profiling_enable();
        [DllImport("loader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void loader_profiling_disable();
        [DllImport("loader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Boolean loader_profiling_set_level(ProfilingLevel level);
        [DllImport("loader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Boolean loader_profiling_save_current_state(string path);
        [DllImport("loader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Boolean loader_profiling_save_snapshots(string path);
        [DllImport("loader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Boolean loader_profiling_save_total_and_snapshots(string path);
        [DllImport("loader.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern string loader_profiling_get_current_state();
        [DllImport("loader.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern string loader_profiling_get_snapshots();
        [DllImport("loader.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        internal static extern string loader_profiling_get_total_and_snapshots();
        [DllImport("loader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void loader_profiling_reset();
        [DllImport("loader.dll", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        internal static extern string loader_get_dll_error();
    }
}
