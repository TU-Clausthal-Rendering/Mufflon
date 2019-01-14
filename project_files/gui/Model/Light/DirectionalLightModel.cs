using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Dll;
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

        private Vec3<float> m_direction;

        public Vec3<float> Direction
        {
            get => m_direction;
            set
            {
                if (Equals(value, m_direction)) return;
                m_direction = value;
                OnPropertyChanged(nameof(Direction));
            }
        }

        private Vec3<float> m_irradiance;

        public Vec3<float> Irradiance
        {
            get => m_irradiance;
            set
            {
                if (Equals(value, m_irradiance)) return;
                m_irradiance = value;
                OnPropertyChanged(nameof(Irradiance));
            }
        }
    }
}
