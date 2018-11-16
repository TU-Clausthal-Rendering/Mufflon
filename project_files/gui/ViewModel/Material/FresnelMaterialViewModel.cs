using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using gui.Command;
using gui.Model;
using gui.Model.Material;
using gui.Utility;
using gui.View.Material;

namespace gui.ViewModel.Material
{
    public class FresnelMaterialViewModel : MaterialViewModel
    {
        private readonly FresnelMaterialModel m_parent;
        private readonly Models m_models;

        public FresnelMaterialViewModel(FresnelMaterialModel parent, Models models) : base(models, parent)
        {
            m_parent = parent;
            m_models = models;
            if (m_parent.LayerRefraction != null)
            {
                var vm = m_parent.LayerRefraction.CreateViewModel(m_models);
                LayerRefraction = vm.CreateView();
            }
            if (m_parent.LayerReflection != null)
            {
                var vm = m_parent.LayerReflection.CreateViewModel(m_models);
                LayerReflection = vm.CreateView();
            }
            AddLayerReflectionCommand = new AddRecursiveMaterialCommand(models, model => parent.LayerReflection = model, model => parent.LayerReflection = null, "Reflection");
            AddLayerRefractionCommand = new AddRecursiveMaterialCommand(models, model => parent.LayerRefraction = model, model => parent.LayerRefraction = null, "Refraction");
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(FresnelMaterialModel.DielectricRefraction):
                    OnPropertyChanged(nameof(RefractionIndex));
                    break;
                case nameof(FresnelMaterialModel.ConductorRefraction):
                    OnPropertyChanged(nameof(RefractionComplexX));
                    OnPropertyChanged(nameof(RefractionComplexY));
                    break;
                case nameof(FresnelMaterialModel.IsDielectric):
                    OnPropertyChanged(nameof(SelectedRefraction));
                    OnPropertyChanged(nameof(DielectricVisibility));
                    OnPropertyChanged(nameof(ConductorVisibility));
                    break;
                case nameof(FresnelMaterialModel.LayerReflection):
                    // create new view
                    if (m_parent.LayerReflection != null)
                    {
                        var vm = m_parent.LayerReflection.CreateViewModel(m_models);
                        LayerReflection = vm.CreateView();
                    }
                    else
                    {
                        LayerReflection = null;
                    }
                    OnPropertyChanged(nameof(LayerReflection));
                    OnPropertyChanged(nameof(LayerReflectionVisibility));
                    OnPropertyChanged(nameof(ButtonReflectionVisibility));
                    break;
                case nameof(FresnelMaterialModel.LayerRefraction):
                    // create new view
                    if (m_parent.LayerRefraction != null)
                    {
                        var vm = m_parent.LayerRefraction.CreateViewModel(m_models);
                        LayerRefraction = vm.CreateView();
                    }
                    else
                    {
                        LayerRefraction = null;
                    }
                    OnPropertyChanged(nameof(LayerRefraction));
                    OnPropertyChanged(nameof(LayerRefractionVisibility));
                    OnPropertyChanged(nameof(ButtonRefractionVisibility));
                    break;
            }
        }

        protected override UIElement CreateInternalView()
        {
            return new FresnelMaterialView(this);
        }

        #region Refraction Index

        public float RefractionIndex
        {
            get => m_parent.DielectricRefraction;
            set => m_parent.DielectricRefraction = value;
        }

        public float RefractionComplexX
        {
            get => m_parent.ConductorRefraction.X;
            set => m_parent.ConductorRefraction = new Vec2<float>(value, m_parent.ConductorRefraction.Y);
        }

        public float RefractionComplexY
        {
            get => m_parent.ConductorRefraction.Y;
            set => m_parent.ConductorRefraction = new Vec2<float>(m_parent.ConductorRefraction.X, value);
        }

        public int SelectedRefraction
        {
            get => m_parent.IsDielectric ? 0 : 1;
            set
            {
                if (value == 0) m_parent.IsDielectric = true;
                if (value == 1) m_parent.IsDielectric = false;
            }
        }

        public Visibility DielectricVisibility => m_parent.IsDielectric ? Visibility.Visible : Visibility.Hidden;

        public Visibility ConductorVisibility => m_parent.IsDielectric ? Visibility.Hidden : Visibility.Visible;

        #endregion

        public object LayerReflection { get; private set; } = null;

        public ICommand AddLayerReflectionCommand { get; }

        public Visibility ButtonReflectionVisibility => LayerReflection == null ? Visibility.Visible : Visibility.Collapsed;

        public Visibility LayerReflectionVisibility => LayerReflection != null ? Visibility.Visible : Visibility.Collapsed;

        public object LayerRefraction { get; private set; } = null;

        public ICommand AddLayerRefractionCommand { get; }

        public Visibility ButtonRefractionVisibility => LayerRefraction == null ? Visibility.Visible : Visibility.Collapsed;

        public Visibility LayerRefractionVisibility => LayerRefraction != null ? Visibility.Visible : Visibility.Collapsed;
    }
}
