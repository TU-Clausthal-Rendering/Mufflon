using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using gui.Annotations;
using gui.Utility;

namespace gui.Model
{

    public class LightModel : INotifyPropertyChanged
    {
        public enum LightType
        {
            Point,
            Directional,
            Spot,
            Envmap,
            Goniometric
        }

        public LightModel(LightType type)
        {
            Type = type;
        }

        public LightType Type { get; }

        private string m_name = String.Empty;

        public string Name
        {
            get => m_name;
            set
            {
                if (value == null || value == m_name) return;
                m_name = value;
                OnPropertyChanged(nameof(Name));
            }
        }

        private float m_scale = 1.0f;

        public float Scale
        {
            get => m_scale;
            set
            {
                Debug.Assert(Scale >= 0.0f);
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_scale) return;
                m_scale = value;
                OnPropertyChanged(nameof(Scale));
            }
        }

        private Vec3<float> m_flux;

        public Vec3<float> Flux
        {
            get
            {
                Debug.Assert(Type == LightType.Point);
                return m_flux;
            }
            set
            {
                Debug.Assert(Type == LightType.Point);
                if (Equals(value, m_flux)) return;
                m_flux = value;
                OnPropertyChanged(nameof(Flux));
            }
        }

        private Vec3<float> m_intensity;

        public Vec3<float> Intensity
        {
            get
            {
                Debug.Assert(Type == LightType.Point || Type == LightType.Spot);
                return m_intensity;
            }
            set
            {
                Debug.Assert(Type == LightType.Point || Type == LightType.Spot);
                if (Equals(value, m_intensity)) return;
                m_intensity = value;
                OnPropertyChanged(nameof(Intensity));
            }
        }

        private Vec3<float> m_position;

        public Vec3<float> Position
        {
            get
            {
                Debug.Assert(Type == LightType.Point || Type == LightType.Spot || Type == LightType.Goniometric);
                return m_position;
            }
            set
            {
                Debug.Assert(Type == LightType.Point || Type == LightType.Spot || Type == LightType.Goniometric);
                if (Equals(m_position, value)) return;
                m_position = value;
                OnPropertyChanged(nameof(Position));
            }
        }

        private Vec3<float> m_direction;

        public Vec3<float> Direction
        {
            get
            {
                Debug.Assert(Type == LightType.Directional || Type == LightType.Spot);
                return m_direction;
            }
            set
            {
                Debug.Assert(Type == LightType.Directional || Type == LightType.Spot);
                if (Equals(value, m_direction)) return;
                m_direction = value;
                OnPropertyChanged(nameof(Direction));
            }
        }

        private Vec3<float> m_radiance;

        public Vec3<float> Radiance
        {
            get
            {
                Debug.Assert(Type == LightType.Directional);
                return m_radiance;
            }
            set
            {
                Debug.Assert(Type == LightType.Directional);
                if (Equals(value, m_radiance)) return;
                m_radiance = value;
                OnPropertyChanged(nameof(Radiance));
            }
        }

        private float m_exponent;

        public float Exponent
        {
            get
            {
                Debug.Assert(Type == LightType.Spot);
                return m_exponent;
            }
            set
            {
                Debug.Assert(Type == LightType.Spot);
                if (Equals(value, m_exponent)) return;
                m_exponent = value;
                OnPropertyChanged(nameof(Exponent));
            }
        }

        private float m_cosWidth;

        public float CosWidth
        {
            get
            {
                Debug.Assert(Type == LightType.Spot);
                return m_cosWidth;
            }
            set
            {
                Debug.Assert(Type == LightType.Spot);
                if (Equals(value, m_cosWidth)) return;
                m_cosWidth = value;
                OnPropertyChanged(nameof(CosWidth));
            }
        }

        private float m_cosFalloffStart;

        public float CosFalloffStart
        {
            get
            {
                Debug.Assert(Type == LightType.Spot);
                return m_cosFalloffStart;
            }
            set
            {
                Debug.Assert(Type == LightType.Spot);
                if (Equals(value, m_cosFalloffStart)) return;
                m_cosFalloffStart = value;
                OnPropertyChanged(nameof(CosFalloffStart));
            }
        }

        private string m_map = String.Empty;

        public string Map
        {
            get
            {
                Debug.Assert(Type == LightType.Envmap || Type == LightType.Goniometric);
                return m_map;
            }
            set
            {
                Debug.Assert(Type == LightType.Envmap || Type == LightType.Goniometric);
                if (Equals(value, m_map)) return;
                m_map = value;
                OnPropertyChanged(nameof(Map));
            }
        }

        #region PropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
}
