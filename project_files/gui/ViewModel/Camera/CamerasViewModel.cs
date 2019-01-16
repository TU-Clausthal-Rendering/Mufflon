using System;
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
            RegisterModelList(models.Cameras);
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
