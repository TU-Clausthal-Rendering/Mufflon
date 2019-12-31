using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Dll;
using gui.Model.Scene;
using gui.Utility;
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

        public EnvmapLightModel(UInt32 handle) : base(handle)
        {
            EnvType = Core.world_get_env_light_type(Handle);
        }

        public Core.BackgroundType EnvType { get; private set; }

        public override float Scale
        {
            get
            {
                if(!Core.world_get_env_light_scale(Handle, out var scale))
                    throw new Exception(Core.core_get_dll_error());
                return scale.x; // TODO!
            }
            set
            {
                if (value == Scale) return;
                if (!Core.world_set_env_light_scale(Handle, new Core.Vec3(value, value, value)))
                    throw new Exception(Core.core_get_dll_error());
                OnPropertyChanged(nameof(Scale));
            }
        }

        /// <summary>
        /// absolute path of the map
        /// </summary>
        public string Map
        {
            get
            {
                if (EnvType != Core.BackgroundType.Envmap) return "";
                var res = Core.world_get_env_light_map(Handle); 
                if(string.IsNullOrEmpty(res))
                    throw new Exception(Core.core_get_dll_error());

                return res;
            }
            set
            {
                if (EnvType != Core.BackgroundType.Envmap || Equals(value, Map)) return;
                
                var texHandle = Core.world_add_texture(value, Core.TextureSampling.Nearest, Core.MipmapType.NONE, null, IntPtr.Zero);
                if(texHandle == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());
                if(!Core.world_set_env_light_map(Handle, texHandle))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Map));
            }
        }

        public float Albedo
        {
            get
            {
                if (EnvType != Core.BackgroundType.SkyHosek) return 0f;
                if (!Core.world_get_sky_light_albedo(Handle, out var res))
                    throw new Exception(Core.core_get_dll_error());
                return res;
            }
            set
            {
                if (value == Albedo || EnvType != Core.BackgroundType.SkyHosek) return;
                if (!Core.world_set_sky_light_albedo(Handle, value))
                    throw new Exception(Core.core_get_dll_error());
                OnPropertyChanged(nameof(Albedo));
            }
        }

        public float SolarRadius
        {
            get
            {
                if (EnvType != Core.BackgroundType.SkyHosek) return 0f;
                if (!Core.world_get_sky_light_solar_radius(Handle, out var res))
                    throw new Exception(Core.core_get_dll_error());
                return res;
            }
            set
            {
                if (value == SolarRadius || EnvType != Core.BackgroundType.SkyHosek) return;
                if (!Core.world_set_sky_light_solar_radius(Handle, value))
                    throw new Exception(Core.core_get_dll_error());
                OnPropertyChanged(nameof(SolarRadius));
            }
        }

        public float Turbidity
        {
            get
            {
                if (EnvType != Core.BackgroundType.SkyHosek) return 0f;
                if (!Core.world_get_sky_light_turbidity(Handle, out var res))
                    throw new Exception(Core.core_get_dll_error());
                return res;
            }
            set
            {
                if (value == Turbidity || EnvType != Core.BackgroundType.SkyHosek) return;
                if (!Core.world_set_sky_light_turbidity(Handle, value))
                    throw new Exception(Core.core_get_dll_error());
                OnPropertyChanged(nameof(Turbidity));
            }
        }

        public Vec3<float> SunDir
        {
            get
            {
                if (EnvType != Core.BackgroundType.SkyHosek) return new Vec3<float>(0f, 0f, 0f);
                if (!Core.world_get_sky_light_sun_direction(Handle, out var res))
                    throw new Exception(Core.core_get_dll_error());
                return res.ToUtilityVec();
            }
            set
            {
                if (Equals(SunDir, value) || EnvType != Core.BackgroundType.SkyHosek) return;
                if (!Core.world_set_sky_light_sun_direction(Handle, new Core.Vec3(value)))
                    throw new Exception(Core.core_get_dll_error());
                OnPropertyChanged(nameof(SunDir));
            }
        }

        public Vec3<float> Color
        {
            get
            {
                if (EnvType != Core.BackgroundType.Monochrome) return new Vec3<float>(0f, 0f, 0f);
                if (!Core.world_get_env_light_color(Handle, out var res))
                    throw new Exception(Core.core_get_dll_error());
                return res.ToUtilityVec();
            }
            set
            {
                if (Equals(value, Color) || EnvType != Core.BackgroundType.Monochrome) return;
                if (!Core.world_set_env_light_color(Handle, new Core.Vec3(value)))
                    throw new Exception(Core.core_get_dll_error());
            }
        }
    }
}
