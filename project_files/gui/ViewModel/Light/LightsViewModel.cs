using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
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
        private readonly Models m_models;

        public LightsViewModel(Models models)
        {
            m_models = models;
            // assume world is zero by default
            Debug.Assert(m_models.World == null);
            m_models.PropertyChanged += ModelsOnPropertyChanged;
        }

        private void ModelsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.World):
                    RegisterModelList(m_models.World?.Lights);
                    break;
            }
        }

        protected override LightViewModel CreateViewModel(LightModel model)
        {
            return model.CreateViewModel(m_models);
        }

        protected override object CreateView(LightViewModel viewModel)
        {
            return viewModel.CreateView();
        }
    }
}
