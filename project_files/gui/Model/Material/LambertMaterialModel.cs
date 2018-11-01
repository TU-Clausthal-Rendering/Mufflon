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
    public class LambertMaterialModel : MaterialModel
    {
        public override MaterialType Type => MaterialType.Lambert;

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
    }
}
