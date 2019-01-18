using gui.Utility;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace gui.Dll
{
    static public class OpenGlDisplay
    {
        [DllImport("opengldisplay.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void opengldisplay_set_gamma(float val);
        [DllImport("opengldisplay.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern float opengldisplay_get_gamma();
        [DllImport("opengldisplay.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void opengldisplay_set_factor(float val);
        [DllImport("opengldisplay.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern float opengldisplay_get_factor();
        [DllImport("opengldisplay.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "opengldisplay_get_dll_error")]
        internal static extern IntPtr opengldisplay_get_dll_error_();
        internal static string opengldisplay_get_dll_error() { return StringUtil.FromNativeUTF8(opengldisplay_get_dll_error_()); }
        [DllImport("opengldisplay.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Boolean opengldisplay_display(int left, int right, int bottom, int top, UInt32 width, UInt32 height);
        [DllImport("opengldisplay.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern UInt32 opengldisplay_get_screen_texture_handle();
        [DllImport("opengldisplay.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Boolean opengldisplay_resize_screen(UInt32 width, UInt32 height, UInt32 format);
        [DllImport("opengldisplay.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Boolean opengldisplay_set_log_level(Core.Severity level);
        [DllImport("opengldisplay.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern Boolean opengldisplay_initialize(Core.LogCallback callback);
        [DllImport("opengldisplay.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void opengldisplay_destroy();

    }
}
