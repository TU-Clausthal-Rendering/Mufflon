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

        public override MaterialViewModel CreateViewModel(Models models)
        {
            return new TorranceMaterialViewModel(models, this);
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

        private bool m_useAlbedoTexture = false;

        public bool UseAlbedoTexture
        {
            get => m_useAlbedoTexture;
            set
            {
                if (value == m_useAlbedoTexture) return;
                m_useAlbedoTexture = value;
                OnPropertyChanged(nameof(UseAlbedoTexture));
            }
        }

        private float m_roughness = 0.5f;
        // isotropic roughness value
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

        private float m_roughnessAnisotropic = 0.0f;

        public float RoughnessAnisotropic
        {
            get => m_roughnessAnisotropic;
            set
            {
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_roughnessAnisotropic) return;
                m_roughnessAnisotropic = value;
                OnPropertyChanged(nameof(RoughnessAnisotropic));
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

        private RoughnessType m_selectedRoughness = RoughnessType.Isotropic;

        public RoughnessType SelectedRoughness
        {
            get => m_selectedRoughness;
            set
            {
                if (value == m_selectedRoughness) return;
                m_selectedRoughness = value;
                OnPropertyChanged(nameof(SelectedRoughness));
            }
        }
    }
}
