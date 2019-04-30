using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Dll;
using gui.Model.Scene;
using gui.Utility;
using gui.ViewModel.Light;

namespace gui.Model.Light
{
    public class PointLightModel : LightModel
    {
        public override LightType Type => LightType.Point;

        public override LightViewModel CreateViewModel(Models models)
        {
            return new PointLightViewModel(models, this);
        }

        public PointLightModel(IntPtr handle) : base(handle)
        {

        }

        public Vec3<float> Position
        {
            get
            {
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_get_point_light_position(Handle, out var pos, frame))
                    throw new Exception(Core.core_get_dll_error());

                return pos.ToUtilityVec();
            }
            set
            {
                if (Equals(Position, value)) return;
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_set_point_light_position(Handle, new Core.Vec3(value), frame))
                    throw new Exception(Core.core_get_dll_error());
                
                OnPropertyChanged(nameof(Position));
            }
        }

        public Vec3<float> Intensity
        {
            get
            {
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_get_point_light_intensity(Handle, out var res, frame))
                    throw new Exception(Core.core_get_dll_error());

                return res.ToUtilityVec();
            }
            set
            {
                if (Equals(value, Intensity)) return;
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_set_point_light_intensity(Handle, new Core.Vec3(value), frame))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Intensity));
            }
        }

        public override uint PathSegments
        {
            get
            {
                uint count;
                if (!Core.world_get_point_light_path_segments(Handle, out count))
                    throw new Exception(Core.core_get_dll_error());
                return count;
            }
        }
    }
}
