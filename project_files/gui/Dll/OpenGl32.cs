using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace gui.Dll
{
    static class OpenGl32
    {
        [DllImport("opengl32.dll", CharSet = CharSet.Auto, SetLastError = true, ExactSpelling = true)]
        public static extern IntPtr wglCreateContext(IntPtr hDC);

        [DllImport("opengl32.dll", CharSet = CharSet.Auto, SetLastError = true, ExactSpelling = true)]
        public static extern bool wglMakeCurrent(IntPtr hDC, IntPtr hRC);


    }
}
