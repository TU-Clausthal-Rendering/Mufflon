using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace gui.Utility
{
    public class Vec2<T>
    {
        public Vec2(T x, T y)
        {
            X = x;
            Y = y;
        }

        public Vec2(T v)
        {
            X = v;
            Y = v;
        }

        public T X { get; set; }
        public T Y { get; set; }
    }
}
