using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace gui.Utility
{
    public struct BoundingBox
    {
        public BoundingBox(Vec3<float> min, Vec3<float> max)
        {
            Min = min;
            Max = max;
        }

        public Vec3<float> Min { get; }
        public Vec3<float> Max { get; }
    }
}
