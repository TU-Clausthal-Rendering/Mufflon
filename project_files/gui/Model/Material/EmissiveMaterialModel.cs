using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Utility;
using gui.ViewModel.Material;

namespace gui.Model.Material
{
    public class EmissiveMaterialModel : MaterialModel
    {
        public override MaterialType Type => MaterialType.Emissive;

        public override MaterialViewModel CreateViewModel(Models models)
        {
            return new EmissiveMaterialViewModel(models, this);
        }

        private Vec3<float> m_radiance;

        public Vec3<float> Radiance
        {
            get => m_radiance;
            set
            {
                if (Equals(m_radiance, value)) return;
                m_radiance = value;
                OnPropertyChanged(nameof(Radiance));
            }
        }

        private string m_radianceTex = String.Empty;

        public string RadianceTex
        {
            get => m_radianceTex;
            set
            {
                if (Equals(m_radianceTex, value)) return;
                m_radianceTex = value;
                OnPropertyChanged(nameof(RadianceTex));
            }
        }

        private bool m_useRadianceTexture = false;

        public bool UseRadianceTexture
        {
            get => m_useRadianceTexture;
            set
            {
                if(value == m_useRadianceTexture) return;
                m_useRadianceTexture = value;
                OnPropertyChanged(nameof(UseRadianceTexture));
            }
        }

        private float m_scale = 1.0f;

        public float Scale
        {
            get => m_scale;
            set
            {
                if (Equals(value, m_scale)) return;
                m_scale = value;
                OnPropertyChanged(nameof(Scale));
            }
        }
    }
}
