using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace gui.Dll
{
    /// <summary>
    /// User32.dll interface
    /// </summary>
    static class User32
    {
        [DllImport("user32.dll", EntryPoint = "CreateWindowEx", CharSet = CharSet.Unicode)]
        internal static extern IntPtr CreateWindowEx(
            int dwExStyle,
            string lpszClassName,
            string lpszWindowName,
            int style,
            int x, int y,
            int width, int height,
            IntPtr hwndParent,
            IntPtr hMenu,
            IntPtr hInst,
            [MarshalAs(UnmanagedType.AsAny)] object pvParam
        );

        [DllImport("user32.dll", EntryPoint = "GetDC", CharSet = CharSet.Unicode)]
        internal static extern IntPtr GetDC(IntPtr hWnd);

        [DllImport("user32.dll", EntryPoint = "DestroyWindow", CharSet = CharSet.Unicode)]
        internal static extern bool DestroyWindow(IntPtr hwnd);

        [Flags]
        internal enum WS_FLAGS : uint
        {
            WS_OVERLAPPED       = 0x00000000,
            WS_TILED            = 0x00000000,
            WS_MAXIMIZEBOX      = 0x00010000,
            WS_TABSTOP          = 0x00010000,
            WS_GROUP            = 0x00020000,
            WS_MINIMIZEBOX      = 0x00020000,
            WS_SIZEBOX          = 0x00040000,
            WS_THICKFRAME       = 0x00040000,
            WS_SYSMENU          = 0x00080000,
            WS_HSCROLL          = 0x00100000,
            WS_VSCROLL          = 0x00200000,
            WS_DLGFRAME         = 0x00400000,
            WS_BORDER           = 0x00800000,
            WS_CAPTION          = 0x00C00000,
            WS_MAXIMIZE         = 0x01000000,
            WS_CLIPCHILDREN     = 0x02000000,
            WS_CLIPSIBLINGS     = 0x04000000,
            WS_DISABLED         = 0x08000000,
            WS_VISIBLE          = 0x10000000,
            WS_ICONIC           = 0x20000000,
            WS_MINIMIZE         = 0x20000000,
            WS_CHILD            = 0x40000000,
            WS_POPUP            = 0x80000000
        }

        [Flags]
        internal enum WS_EX_FLAGS : uint
        {
            WS_EX_DLGMODALFRAME         = 0x00000001,
            WS_EX_NOPARENTNOTIFY        = 0x00000004,
            WS_EX_TOPMOST               = 0x00000008,
            WS_EX_ACCEPTFILES           = 0x00000010,
            WS_EX_TRANSPARENT           = 0x00000020,
            WS_EX_MDICHILD              = 0x00000040,
            WS_EX_TOOLWINDOW            = 0x00000080,
            WS_EX_WINDOWEDGE            = 0x00000100,
            WS_EX_CLIENTEDGE            = 0x00000200,
            WS_EX_CONTEXTHELP           = 0x00000400,
            WS_EX_RIGHT                 = 0x00001000,
            WS_EX_RTLREADING            = 0x00002000,
            WS_EX_LEFTSCROLLBAR         = 0x00004000,
            WS_EX_CONTROLPARENT         = 0x00010000,
            WS_EX_STATICEDGE            = 0x00020000,
            WS_EX_APPWINDOW             = 0x00040000,
            WS_EX_LAYERED               = 0x00080000,
            WS_EX_NOINHERITLAYOUT       = 0x00100000,
            WS_EX_NOREDIRECTIONBITMAP   = 0x00200000,
            WS_EX_LAYOUTRTL             = 0x00400000,
            WS_EX_COMPOSITED            = 0x02000000,
            WS_EX_NOACTIVATE            = 0x08000000
        }
    }
}
