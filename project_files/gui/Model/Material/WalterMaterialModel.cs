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
    public class WalterMaterialModel : MaterialModel
    {
        public override MaterialType Type => MaterialType.Walter;

        public override MaterialViewModel CreateViewModel(Models models)
        {
            throw new NotImplementedException();
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

        private Vec3<float> m_absorption;

        public Vec3<float> Absorption
        {
            get => m_absorption;
            set
            {
                if (Equals(value, m_absorption)) return;
                m_absorption = value;
                OnPropertyChanged(nameof(Absorption));
            }
        }
    }
}
