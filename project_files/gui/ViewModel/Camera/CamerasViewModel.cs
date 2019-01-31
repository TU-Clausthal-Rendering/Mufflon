using System;
using System.ComponentModel;
using System.Diagnostics;
using gui.Model;
using gui.Model.Camera;
using gui.Utility;

namespace gui.ViewModel.Camera
{
    public class CamerasViewModel : SynchronizedViewModelList<CameraModel, CameraViewModel, object>
    {
        private readonly Models m_models;

        public CamerasViewModel(Models models)
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
                    RegisterModelList(m_models.World?.Cameras);
                    break;
            }
        }

        protected override CameraViewModel CreateViewModel(CameraModel model)
        {
            return model.CreateViewModel(m_models);
        }

        protected override object CreateView(CameraViewModel viewModel)
        {
            return viewModel.CreateView();
        }
    }
}
