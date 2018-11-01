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
    public class TorranceMaterialModel : MaterialModel
    {
        public override MaterialType Type => MaterialType.Torrance;

        public override MaterialViewModel CreateViewModel()
        {
            throw new NotImplementedException();
        }

        private Vec3<float> m_albedo = new Vec3<float>(0.5f);

        public Vec3<float> Albedo
        {
            get => m_albedo;
            set
            {
                if (Equals(value, m_albedo)) return;
                m_albedo = value;
                OnPropertyChanged(nameof(Albedo));
            }
        }

        private string m_albedoTex = String.Empty;

        public string AlbedoTex
        {
            get => m_albedoTex;
            set
            {
                if (Equals(value, m_albedoTex)) return;
                m_albedoTex = value;
                OnPropertyChanged(nameof(AlbedoTex));
            }
        }

        private float m_roughness = 0.5f;
        // isotrophic roughness value
        public float Roughness
        {
            get => m_roughness;
            set
            {
                if (Equals(value, m_roughness)) return;
                m_roughness = value;
                OnPropertyChanged(nameof(Roughness));
            }
        }

        private float m_roughnessAngleX;

        public float RoughnessAngleX
        {
            get => m_roughnessAngleX;
            set
            {
                if (Equals(value, m_roughnessAngleX)) return;
                m_roughnessAngleX = value;
                OnPropertyChanged(nameof(RoughnessAngleX));
            }
        }

        private float m_roughnessAngleY;

        public float RoughnessAngleY
        {
            get => m_roughnessAngleY;
            set
            {
                if (Equals(value, m_roughnessAngleY)) return;
                m_roughnessAngleY = value;
                OnPropertyChanged(nameof(RoughnessAngleY));
            }
        }

        private string m_roughnessTex = String.Empty;

        public string RoughnessTex
        {
            get => m_roughnessTex;
            set
            {
                if (Equals(value, m_roughnessTex)) return;
                m_roughnessTex = value;
                OnPropertyChanged(nameof(RoughnessTex));
            }
        }
    }
}
