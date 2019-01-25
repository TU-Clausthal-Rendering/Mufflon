using System;
using System.Runtime.InteropServices;

namespace gui.Dll
{
    static class OpenGl32
    {
        public delegate IntPtr WglCreateContextAttribsARB(IntPtr hDC, IntPtr hShareContext, int[] attribList);
        public delegate IntPtr WglGetExtensionsStringARB(IntPtr hDC);
        private static WglGetExtensionsStringARB wglGetExtensionsStringARB = null;

        [DllImport("opengl32.dll", EntryPoint = "glGetString", CharSet = CharSet.Auto, SetLastError = true, ExactSpelling = true)]
        private static extern IntPtr _glGetString(StringName name);

        [DllImport("opengl32.dll", CharSet = CharSet.Auto, SetLastError = true, ExactSpelling = true)]
        public static extern IntPtr wglCreateContext(IntPtr hDC);

        [DllImport("opengl32.dll", CharSet = CharSet.Auto, SetLastError = true, ExactSpelling = true, EntryPoint = "wglGetProcAddress")]
        private static extern IntPtr _GetProcAddress(string name);

        [DllImport("opengl32.dll", CharSet = CharSet.Auto, SetLastError = true, ExactSpelling = true)]
        public static extern bool wglDeleteContext(IntPtr hDC);

        [DllImport("opengl32.dll", CharSet = CharSet.Auto, SetLastError = true, ExactSpelling = true)]
        public static extern bool wglMakeCurrent(IntPtr hDC, IntPtr hRC);

        [DllImport("opengl32.dll", CharSet = CharSet.Auto, SetLastError = true, ExactSpelling = true)]
        public static extern void glViewport(int offsetX, int offsetY, UInt32 width, UInt32 height);

        public static string glGetString(StringName name)
        {
            return Marshal.PtrToStringAnsi(_glGetString(name));
        }

        public static string wglGetExtensionsString(IntPtr hDC)
        {
            if (wglGetExtensionsStringARB == null)
                wglGetExtensionsStringARB = wglGetProcAddress<WglGetExtensionsStringARB>("wglGetExtensionsStringARB");
            if (wglGetExtensionsStringARB != null)
                return Marshal.PtrToStringAnsi(wglGetExtensionsStringARB(hDC));
            return "";
        }

        public static TDelegate wglGetProcAddress<TDelegate>(string name) where TDelegate : class
        {
            IntPtr addr = _GetProcAddress(name);
            if (addr == IntPtr.Zero)
                return null;
            return (TDelegate)((object)Marshal.GetDelegateForFunctionPointer(addr, typeof(TDelegate)));
        }

        public enum WglContextAttributeNames
        {
            CONTEXT_MAJOR_VERSION_ARB = 0x2091,
            CONTEXT_MINOR_VERSION_ARB = 0x2092,
            CONTEXT_LAYER_PLANE_ARB = 0x2093,
            CONTEXT_FLAGS_ARB = 0x2094,
            CONTEXT_PROFILE_MASK_ARB = 0x9126
        };

        public enum WglContextFlags
        {
            NONE_BIT = 0x0,
            CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT = 0x00000001,
            CONTEXT_FLAG_DEBUG_BIT = 0x00000002,
            CONTEXT_FLAG_ROBUST_ACCESS_BIT = 0x00000004,
            CONTEXT_FLAG_NO_ERROR_BIT = 0x00000008
        };

        public enum WglContextProfileFlags
        {
            NONE_BIT = 0x0,
            WGL_CONTEXT_CORE_PROFILE_BIT = 0x0001,
            WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT = 0x0002,
        };

        public enum StringName
        {
            VENDOR = 0x1F00,
            RENDERER = 0x1F01,
            VERSION = 0x1F02,
            EXTENSIONS = 0x1F03
        };
    }
}
