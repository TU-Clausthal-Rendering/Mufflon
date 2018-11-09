using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.ViewModel.Material;

namespace gui.Model.Material
{
    public class BlendMaterialModel : MaterialModel
    {
        public override MaterialType Type => MaterialType.Blend;

        public override MaterialViewModel CreateViewModel(Models models)
        {
            return new BlendMaterialViewModel(models, this);
        }

        private MaterialModel m_layerA = null;

        public MaterialModel LayerA
        {
            get => m_layerA;
            set
            {
                if (ReferenceEquals(value, m_layerA)) return;
                m_layerA = value;
                OnPropertyChanged(nameof(LayerA));
            }
        }

        private MaterialModel m_layerB = null;

        public MaterialModel LayerB
        {
            get => m_layerB;
            set
            {
                if (ReferenceEquals(value, m_layerB)) return;
                m_layerB = value;
                OnPropertyChanged(nameof(LayerB));
            }
        }

        private float m_factorA;

        public float FactorA
        {
            get => m_factorA;
            set
            {
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_factorA) return;
                m_factorA = value;
                OnPropertyChanged(nameof(FactorA));
            }
        }

        private float m_factorB;

        public float FactorB
        {
            get => m_factorB;
            set
            {
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_factorB) return;
                m_factorB = value;
                OnPropertyChanged(nameof(FactorB));
            }
        }


        public BlendMaterialModel(bool isRecursive, Action<MaterialModel> removeAction) : base(isRecursive, removeAction)
        {
        }
    }
}
