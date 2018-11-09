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

        public override MaterialViewModel CreateViewModel(Models models)
        {
            return new LambertMaterialViewModel(models, this);
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

        public LambertMaterialModel(bool isRecursive, Action<MaterialModel> removeAction) : base(isRecursive, removeAction)
        {
        }
    }
}
