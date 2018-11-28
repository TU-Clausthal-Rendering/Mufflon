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
        [DllImport("loader.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern bool loader_load_json(string path);
    }
}
