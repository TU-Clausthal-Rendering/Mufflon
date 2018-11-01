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
        public MaterialsViewModel(Models models) : base(models.Materials)
        {

        }

        protected override MaterialViewModel CreateViewModel(MaterialModel model)
        {
            return model.CreateViewModel();
        }

        protected override object CreateView(MaterialViewModel viewModel)
        {
            return viewModel.CreateView();
        }
    }
}
