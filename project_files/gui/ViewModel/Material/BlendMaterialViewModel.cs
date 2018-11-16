using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using gui.Command;
using gui.Model;
using gui.Model.Material;
using gui.View.Material;

namespace gui.ViewModel.Material
{
    public class BlendMaterialViewModel : MaterialViewModel
    {
        private readonly BlendMaterialModel m_parent;
        private readonly Models m_models;

        public BlendMaterialViewModel(Models models, BlendMaterialModel parent) : base(models, parent)
        {
            m_parent = parent;
            m_models = models;
            if (m_parent.LayerA != null)
            {
                var vm = m_parent.LayerA.CreateViewModel(models);
                LayerA = vm.CreateView();
            }
            if (m_parent.LayerB != null)
            {
                var vm = m_parent.LayerB.CreateViewModel(models);
                LayerB = vm.CreateView();
            }
            AddLayerACommand = new AddRecursiveMaterialCommand(models, model => parent.LayerA = model, model => parent.LayerA = null, "LayerA");
            AddLayerBCommand = new AddRecursiveMaterialCommand(models, model => parent.LayerB = model, model => parent.LayerB = null, "LayerB");
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
                        var vm = m_parent.LayerA.CreateViewModel(m_models);
                        LayerA = vm.CreateView();
                    }
                    else
                    {
                        LayerA = null;
                    }
                    OnPropertyChanged(nameof(LayerA));
                    OnPropertyChanged(nameof(LayerAVisibility));
                    OnPropertyChanged(nameof(ButtonAVisibility));
                    break;
                case nameof(BlendMaterialModel.LayerB):
                    if (m_parent.LayerB != null)
                    {
                        var vm = m_parent.LayerB.CreateViewModel(m_models);
                        LayerB = vm.CreateView();
                    }
                    else
                    {
                        LayerB = null;
                    }
                    OnPropertyChanged(nameof(LayerB));
                    OnPropertyChanged(nameof(LayerBVisibility));
                    OnPropertyChanged(nameof(ButtonBVisibility));
                    break;
            }
        }

        protected override UIElement CreateInternalView()
        {
            return new BlendMaterialView(this);
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

        public ICommand AddLayerACommand { get; }

        public Visibility ButtonAVisibility => LayerA == null ? Visibility.Visible : Visibility.Collapsed;

        public Visibility LayerAVisibility => LayerA != null ? Visibility.Visible : Visibility.Collapsed;

        public object LayerB { get; private set; } = null;

        public ICommand AddLayerBCommand { get; }

        public Visibility ButtonBVisibility => LayerB == null ? Visibility.Visible : Visibility.Collapsed;

        public Visibility LayerBVisibility => LayerB != null ? Visibility.Visible : Visibility.Collapsed;
    }
}
