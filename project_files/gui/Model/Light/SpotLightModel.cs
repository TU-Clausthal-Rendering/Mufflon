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
    public class SpotLightModel : LightModel
    {
        public override LightType Type => LightType.Spot;

        public override LightViewModel CreateViewModel(Models models)
        {
            return new SpotLightViewModel(models, this);
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

        private float m_exponent;

        public float Exponent
        {
            get => m_exponent;
            set
            {
                if (Equals(value, m_exponent)) return;
                m_exponent = value;
                OnPropertyChanged(nameof(Exponent));
            }
        }

        private float m_width;

        public float Width
        {
            get => m_width;
            set
            {
                if (Equals(value, m_width)) return;
                m_width = value;
                OnPropertyChanged(nameof(Width));
            }
        }

        private float m_falloffStart;

        public float FalloffStart
        {
            get
            {
                Debug.Assert(Type == LightType.Spot);
                return m_falloffStart;
            }
            set
            {
                Debug.Assert(Type == LightType.Spot);
                if (Equals(value, m_falloffStart)) return;
                m_falloffStart = value;
                OnPropertyChanged(nameof(FalloffStart));
            }
        }
    }
}
