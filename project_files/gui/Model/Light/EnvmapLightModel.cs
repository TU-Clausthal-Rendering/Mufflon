using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Dll;
using gui.Model.Scene;
using gui.ViewModel.Light;

namespace gui.Model.Light
{
    public class EnvmapLightModel : LightModel
    {
        public override LightType Type => LightType.Envmap;

        public override LightViewModel CreateViewModel(Models models)
        {
            return new EnvmapLightViewModel(models, this);
        }

        public EnvmapLightModel(IntPtr handle) : base(handle)
        {
            
        }

        /// <summary>
        /// absolute path of the map
        /// </summary>
        public string Map
        {
            get
            {
                var res = Core.world_get_env_light_map(Handle); 
                if(string.IsNullOrEmpty(res))
                    throw new Exception(Core.core_get_dll_error());

                return res;
            }
            set
            {
                if (Equals(value, Map)) return;
                //var absolutePath = Path.Combine(m_world.Directory, value);
                var texHandle = Core.world_add_texture(value, Core.TextureSampling.LINEAR);
                if(texHandle == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());
                if(!Core.world_set_env_light_map(Handle, texHandle))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Map));
            }
        }
    }
}
