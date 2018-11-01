using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model;
using gui.Model.Light;
using gui.Utility;

namespace gui.ViewModel.Light
{
    public class LightsViewModel : SynchronizedViewModelList<LightModel, LightViewModel, object>
    {
        public LightsViewModel(Models models) : base(models.Lights)
        {
        }

        protected override LightViewModel CreateViewModel(LightModel model)
        {
            return model.CreateViewModel();
        }

        protected override object CreateView(LightViewModel viewModel)
        {
            return viewModel.CreateView();
        }
    }
}
