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
            return new FresnelMaterialViewModel(this, models);
        }

        private float m_dielecticRefraction;

        public float DielectricRefraction
        {
            get => m_dielecticRefraction;
            set
            {
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_dielecticRefraction) return;
                m_dielecticRefraction = value;
                OnPropertyChanged(nameof(DielectricRefraction));
            }
        }

        private Vec2<float> m_conductorRefraction;

        public Vec2<float> ConductorRefraction
        {
            get => m_conductorRefraction;
            set
            {
                if (Equals(value, m_conductorRefraction)) return;
                m_conductorRefraction = value;
                OnPropertyChanged(nameof(ConductorRefraction));
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

        private bool m_isDielectric = true;

        public bool IsDielectric
        {
            get => m_isDielectric;
            set
            {
                if(value == m_isDielectric) return;
                m_isDielectric = value;
                OnPropertyChanged(nameof(IsDielectric));
            }
        }

        public FresnelMaterialModel(bool isRecursive, Action<MaterialModel> removeAction) : base(isRecursive, removeAction)
        {
        }
    }
}
