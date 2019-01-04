
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace gui.Utility
{
    class StringUtil
    {
        // Convert a "const char*" from the c methods to a usable string object.
        // From https://stackoverflow.com/questions/10773440/conversion-in-net-native-utf-8-managed-string
        internal static string FromNativeUTF8(IntPtr nativeUtf8)
        {
            int len = 0;
            while (Marshal.ReadByte(nativeUtf8, len) != 0) ++len;
            byte[] buffer = new byte[len];
            Marshal.Copy(nativeUtf8, buffer, 0, buffer.Length);
            return Encoding.UTF8.GetString(buffer);
        }

        internal static IntPtr ToNativeUtf8(string managedString)
        {
            byte[] buffer = Encoding.UTF8.GetBytes(managedString); // not null terminated
            Array.Resize(ref buffer, buffer.Length + 1);
            buffer[buffer.Length - 1] = 0; // terminating 0
            IntPtr nativeUtf8 = Marshal.AllocHGlobal(buffer.Length);
            Marshal.Copy(buffer, 0, nativeUtf8, buffer.Length);
            return nativeUtf8;
        }
    }
}
