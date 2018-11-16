using System;
using System.Collections.Generic;
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

        public MaterialsViewModel(Models models) : base(models.Materials)
        {
            m_models = models;
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
