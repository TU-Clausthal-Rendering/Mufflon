using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model;
using gui.Model.Material;
using gui.Utility;

namespace gui.ViewModel.Material
{
    public class MaterialsViewModel : SynchronizedViewModelList<MaterialModel, MaterialViewModel, object>
    {
        private readonly Models m_models;

        public MaterialsViewModel(Models models)
        {
            m_models = models;
            Debug.Assert(m_models.World == null);
            m_models.PropertyChanged += ModelsOnPropertyChanged;
        }

        private void ModelsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.World):
                    RegisterModelList(m_models.World?.Materials);
                    break;
            }
        }

        protected override MaterialViewModel CreateViewModel(MaterialModel model)
        {
            return model.CreateViewModel(m_models);
        }

        protected override object CreateView(MaterialViewModel viewModel)
        {
            return viewModel.CreateView();
        }
    }
}
