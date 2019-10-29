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
    public class SpotLightModel : LightModel
    {
        public override LightType Type => LightType.Spot;

        public override LightViewModel CreateViewModel(Models models)
        {
            return new SpotLightViewModel(models, this);
        }

        public SpotLightModel(IntPtr handle) : base(handle)
        {

        }

        public override float Scale
        {
            get => 1f;
            set
            {
                if (value == Scale) return;
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                var intensity = Intensity;
                Intensity = new Vec3<float>(intensity.X * value, intensity.Y * value, intensity.Z * value);
                OnPropertyChanged(nameof(Scale));
            }
        }

        public Vec3<float> Position
        {
            get
            {
                uint frame;
                if(!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if(!Core.world_get_spot_light_position(Handle, out var res, frame))
                    throw new Exception(Core.core_get_dll_error());

                return res.ToUtilityVec();
            }
            set
            {
                if (Equals(Position, value)) return;
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_set_spot_light_position(Handle, new Core.Vec3(value), frame))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Position));
            }
        }

        public Vec3<float> Direction
        {
            get
            {
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_get_spot_light_direction(Handle, out var res, frame))
                    throw new Exception(Core.core_get_dll_error());

                return res.ToUtilityVec();
            }
            set
            {
                if (Equals(Direction, value)) return;
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_set_spot_light_direction(Handle, new Core.Vec3(value), frame))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Direction));
            }
        }

        public Vec3<float> Intensity
        {
            get
            {
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_get_spot_light_intensity(Handle, out var res, frame))
                    throw new Exception(Core.core_get_dll_error());

                return res.ToUtilityVec();
            }
            set
            {
                if (Equals(Intensity, value)) return;
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_set_spot_light_intensity(Handle, new Core.Vec3(value), frame))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Intensity));
            }
        }

        public float Width
        {
            get
            {
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_get_spot_light_angle(Handle, out var res, frame))
                    throw new Exception(Core.core_get_dll_error());

                return res;
            }
            set
            {
                if (Equals(Width, value)) return;
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_set_spot_light_angle(Handle, value, frame))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Width));
            }
        }

        public float Falloff
        {
            get
            {
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_get_spot_light_falloff(Handle, out var res, frame))
                    throw new Exception(Core.core_get_dll_error());

                return res;
            }
            set
            {
                if (Equals(Falloff, value)) return;
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_set_spot_light_falloff(Handle, value, frame))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Falloff));
            }
        }

        public override uint PathSegments
        {
            get
            {
                uint count;
                if (!Core.world_get_spot_light_path_segments(Handle, out count))
                    throw new Exception(Core.core_get_dll_error());
                return count;
            }
        }
    }
}
