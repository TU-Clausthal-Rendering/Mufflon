using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using gui.Model.Material;

namespace gui.ViewModel.Material
{
    public class BlendMaterialViewModel : MaterialViewModel
    {
        private readonly BlendMaterialModel m_parent;

        public BlendMaterialViewModel(BlendMaterialModel parent) : base(parent)
        {
            m_parent = parent;
            if (m_parent.LayerA != null)
            {
                var vm = m_parent.LayerA.CreateViewModel();
                LayerA = vm.CreateView();
            }
            if (m_parent.LayerB != null)
            {
                var vm = m_parent.LayerB.CreateViewModel();
                LayerB = vm.CreateView();
            }
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(BlendMaterialModel.FactorA):
                    OnPropertyChanged(nameof(FactorA));
                    break;
                case nameof(BlendMaterialModel.FactorB):
                    OnPropertyChanged(nameof(FactorB));
                    break;
                case nameof(BlendMaterialModel.LayerA):
                    if (m_parent.LayerA != null)
                    {
                        var vm = m_parent.LayerA.CreateViewModel();
                        LayerA = vm.CreateView();
                    }
                    else
                    {
                        LayerA = null;
                    }
                    OnPropertyChanged(nameof(LayerA));
                    break;
                case nameof(BlendMaterialModel.LayerB):
                    if (m_parent.LayerB != null)
                    {
                        var vm = m_parent.LayerB.CreateViewModel();
                        LayerB = vm.CreateView();
                    }
                    else
                    {
                        LayerB = null;
                    }
                    OnPropertyChanged(nameof(LayerB));
                    break;
            }
        }

        public override object CreateView()
        {
            throw new NotImplementedException();
        }

        public float FactorA
        {
            get => m_parent.FactorA;
            set => m_parent.FactorA = value;
        }

        public float FactorB
        {
            get => m_parent.FactorB;
            set => m_parent.FactorB = value;
        }

        public object LayerA { get; private set; } = null;

        public object LayerB { get; private set; } = null;
    }
}
