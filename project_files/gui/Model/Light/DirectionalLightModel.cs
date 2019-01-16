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

        public DirectionalLightModel(IntPtr handle, WorldModel world) : base(handle, world)
        {

        }

        public Vec3<float> Direction
        {
            get
            {
                if(!Core.world_get_dir_light_direction(Handle, out var res))
                    throw new Exception(Core.core_get_dll_error());

                return res.ToUtilityVec();
            }
            set
            {
                if (Equals(value, Direction)) return;
                if(!Core.world_set_dir_light_direction(Handle, new Core.Vec3(value)))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Direction));
            }
        }

        public Vec3<float> Irradiance
        {
            get
            {
                if(!Core.world_get_dir_light_irradiance(Handle, out var res))
                    throw new Exception(Core.core_get_dll_error());

                return res.ToUtilityVec();
            }
            set
            {
                if (Equals(value, Irradiance)) return;
                if(!Core.world_set_dir_light_irradiance(Handle, new Core.Vec3(value)))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Irradiance));
            }
        }
    }
}
