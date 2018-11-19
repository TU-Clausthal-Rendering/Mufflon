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
        public delegate void LogCallback(string message, int severity);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool iterate();

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool mufflon_initialize(LogCallback logCallback);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool mufflon_destroy();

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool resize(int width, int height, int offsetX, int offsetY);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr get_error(out int length);

        [DllImport("core.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void execute_command(string command);

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
