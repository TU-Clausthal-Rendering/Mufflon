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
    public class FresnelMaterialModel : MaterialModel
    {
        public override MaterialType Type => MaterialType.Fresnel;

        public override MaterialViewModel CreateViewModel(Models models)
        {
            throw new NotImplementedException();
        }

        private float m_refractionIndex;

        public float RefractionIndex
        {
            get => m_refractionIndex;
            set
            {
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_refractionIndex) return;
                m_refractionIndex = value;
                OnPropertyChanged(nameof(RefractionIndex));
            }
        }

        private Vec2<float> m_refractionComplex;

        public Vec2<float> RefractionComplex
        {
            get => m_refractionComplex;
            set
            {
                if (Equals(value, m_refractionComplex)) return;
                m_refractionComplex = value;
                OnPropertyChanged(nameof(RefractionComplex));
            }
        }

        private MaterialModel m_layerReflection = null;

        public MaterialModel LayerReflection
        {
            get => m_layerReflection;
            set
            {
                if (ReferenceEquals(value, m_layerReflection)) return;
                m_layerReflection = value;
                OnPropertyChanged(nameof(LayerReflection));
            }
        }

        private MaterialModel m_layerRefraction = null;

        public MaterialModel LayerRefraction
        {
            get => m_layerRefraction;
            set
            {
                if (ReferenceEquals(value, m_layerRefraction)) return;
                m_layerRefraction = value;
                OnPropertyChanged(nameof(LayerRefraction));
            }
        }
    }
}
