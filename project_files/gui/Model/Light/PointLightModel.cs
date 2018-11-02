using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Utility;
using gui.ViewModel.Light;

namespace gui.Model.Light
{
    public class PointLightModel : LightModel
    {
        public override LightType Type => LightType.Point;

        public override LightViewModel CreateViewModel(Models models)
        {
            return new PointLightViewModel(this);
        }

        private Vec3<float> m_position;

        public Vec3<float> Position
        {
            get => m_position;
            set
            {
                if (Equals(m_position, value)) return;
                m_position = value;
                OnPropertyChanged(nameof(Position));
            }
        }

        private Vec3<float> m_intensity;

        public Vec3<float> Intensity
        {
            get => m_intensity;
            set
            {
                if (Equals(value, m_intensity)) return;
                m_intensity = value;
                OnPropertyChanged(nameof(Intensity));
            }
        }
    }
}
