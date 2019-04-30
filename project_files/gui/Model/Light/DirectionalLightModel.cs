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
    public class DirectionalLightModel : LightModel
    {
        public override LightType Type => LightType.Directional;

        public override LightViewModel CreateViewModel(Models models)
        {
            return new DirectionalLightViewModel(models, this);
        }

        public DirectionalLightModel(IntPtr handle) : base(handle)
        {

        }

        public Vec3<float> Direction
        {
            get
            {
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_get_dir_light_direction(Handle, out var res, frame))
                    throw new Exception(Core.core_get_dll_error());

                return res.ToUtilityVec();
            }
            set
            {
                if (Equals(value, Direction)) return;
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_set_dir_light_direction(Handle, new Core.Vec3(value), frame))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Direction));
            }
        }

        public Vec3<float> Irradiance
        {
            get
            {
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_get_dir_light_irradiance(Handle, out var res, frame))
                    throw new Exception(Core.core_get_dll_error());

                return res.ToUtilityVec();
            }
            set
            {
                if (Equals(value, Irradiance)) return;
                uint frame;
                if (!Core.world_get_frame_current(out frame))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_set_dir_light_irradiance(Handle, new Core.Vec3(value), frame))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Irradiance));
            }
        }

        public override uint PathSegments
        {
            get
            {
                uint count;
                if (!Core.world_get_dir_light_path_segments(Handle, out count))
                    throw new Exception(Core.core_get_dll_error());
                return count;
            }
        }
    }
}
