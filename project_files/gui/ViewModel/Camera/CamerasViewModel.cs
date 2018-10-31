using System;
using gui.Model;
using gui.Model.Camera;
using gui.Utility;

namespace gui.ViewModel.Camera
{
    public class CamerasViewModel : SynchronizedViewModelList<CameraModel, CameraViewModel, object>
    {
        public CamerasViewModel(Models models) : base(models.Cameras)
        {

        }

        protected override CameraViewModel CreateViewModel(CameraModel model)
        {
            return model.CreateViewModel();
        }

        protected override object CreateView(CameraViewModel viewModel)
        {
            return viewModel.CreateView();
        }
    }
}
