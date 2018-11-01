using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model.Material;

namespace gui.ViewModel.Material
{
    public class FresnelMaterialViewModel : MaterialViewModel
    {
        private readonly FresnelMaterialModel m_parent;

        public FresnelMaterialViewModel(FresnelMaterialModel parent) : base(parent)
        {
            m_parent = parent;
            if (m_parent.LayerRefraction != null)
            {
                var vm = m_parent.LayerRefraction.CreateViewModel();
                LayerRefraction = vm.CreateView();
            }
            if (m_parent.LayerReflection != null)
            {
                var vm = m_parent.LayerReflection.CreateViewModel();
                LayerReflection = vm.CreateView();
            }
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(FresnelMaterialModel.RefractionIndex):
                    OnPropertyChanged(nameof(RefractionIndex));
                    break;
                case nameof(FresnelMaterialModel.RefractionComplex):

                    break;
                case nameof(FresnelMaterialModel.LayerReflection):
                    // create new view
                    if (m_parent.LayerReflection != null)
                    {
                        var vm = m_parent.LayerReflection.CreateViewModel();
                        LayerReflection = vm.CreateView();
                    }
                    else
                    {
                        LayerReflection = null;
                    }
                    OnPropertyChanged(nameof(LayerReflection));
                    break;
                case nameof(FresnelMaterialModel.LayerRefraction):
                    // create new view
                    if (m_parent.LayerRefraction != null)
                    {
                        var vm = m_parent.LayerRefraction.CreateViewModel();
                        LayerRefraction = vm.CreateView();
                    }
                    else
                    {
                        LayerRefraction = null;
                    }
                    OnPropertyChanged(nameof(LayerRefraction));
                    break;
            }
        }

        public override object CreateView()
        {
            throw new NotImplementedException();
        }

        public float RefractionIndex
        {
            get => m_parent.RefractionIndex;
            set => m_parent.RefractionIndex = value;
        }

        // TODO Refraction Complex?

        public object LayerReflection { get; private set; } = null;

        public object LayerRefraction { get; private set; } = null;
    }
}
