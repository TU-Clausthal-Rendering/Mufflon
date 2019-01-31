using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Dll;
using gui.Utility;

namespace gui.Model.Light
{
    public class LightsModel : SynchronizedModelList<LightModel>
    {
        // Loads lights for the current world
        public LightsModel()
        {
            var numPointLights = Core.world_get_point_light_count();
            for (var i = 0u; i < numPointLights; ++i)
                m_list.Add(new PointLightModel(Core.world_get_light_handle(i, Core.LightType.POINT)));

            var numSpotLights = Core.world_get_spot_light_count();
            for (var i = 0u; i < numSpotLights; ++i)
                m_list.Add(new SpotLightModel(Core.world_get_light_handle(i, Core.LightType.SPOT)));

            var numDirLights = Core.world_get_dir_light_count();
            for (var i = 0u; i < numDirLights; ++i)
                m_list.Add(new DirectionalLightModel(Core.world_get_light_handle(i, Core.LightType.DIRECTIONAL)));

            var numEnvLights = Core.world_get_env_light_count();
            for (var i = 0u; i < numEnvLights; ++i)
                m_list.Add(new EnvmapLightModel(Core.world_get_light_handle(i, Core.LightType.ENVMAP)));
        }

        public void AddLight(string name, LightModel.LightType type)
        {
            var handle = Core.world_add_light(name, Core.FromModelLightType(type));
            var lm = LightModel.MakeFromHandle(handle, type);
            m_list.Add(lm);
        }
    }
}
