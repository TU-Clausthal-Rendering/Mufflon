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

        public T X { get; }
        public T Y { get; }
        public T Z { get; }
    }
}
