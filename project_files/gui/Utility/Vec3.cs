using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace gui.Utility
{
    public struct Vec3<T>
    {
        public Vec3(T x, T y, T z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public Vec3(T v)
        {
            X = v;
            Y = v;
            Z = v;
        }

        public T X { get; set; }
        public T Y { get; set; }
        public T Z { get; set; }
    }

    public struct Vec4<T>
    {
        public Vec4(T x, T y, T z, T w)
        {
            X = x;
            Y = y;
            Z = z;
            W = w;
        }

        public Vec4(T v)
        {
            X = v;
            Y = v;
            Z = v;
            W = v;
        }

        public static bool operator==(Vec4<T> v1, Vec4<T> v2)
        {
            return v1.X.Equals(v2.X) && v1.Y.Equals(v2.Y) && v1.Z.Equals(v2.Z) && v1.W.Equals(v2.W);
        }

        public static bool operator!=(Vec4<T> v1, Vec4<T> v2)
        {
            return !(v1.X.Equals(v2.X) && v1.Y.Equals(v2.Y) && v1.Z.Equals(v2.Z) && v1.W.Equals(v2.W));
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            return obj.GetType() == GetType() && ((Vec4<T>)obj) == this;
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public override string ToString()
        {
            return "[" + X.ToString() + "|" + Y.ToString() + "|" + Z.ToString() + "|" + W.ToString() + "]";
        }

        public T X { get; set; }
        public T Y { get; set; }
        public T Z { get; set; }
        public T W { get; set; }
    }
}
