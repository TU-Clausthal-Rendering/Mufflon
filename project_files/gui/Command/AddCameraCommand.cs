using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Model;
using gui.Model.Camera;
using gui.View.Dialog;
using gui.ViewModel.Camera;

namespace gui.Command
{
    public class AddCameraCommand : ICommand
    {
        private readonly Models m_models;

        public AddCameraCommand(Models models)
        {
            m_models = models;
        }

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public void Execute(object parameter)
        {
            var dc = new AddCameraViewModel();
            var dialog = new AddPropertyDialog(dc);

            if (dialog.ShowDialog() != true) return;

            CameraModel cm = null;
            switch (dc.TypeValue)
            {
                case CameraModel.CameraType.Pinhole:
                    cm = new PinholeCameraModel();
                    break;
                case CameraModel.CameraType.Focus:
                    cm = new FocusCameraModel();
                    break;
                case CameraModel.CameraType.Ortho:
                    cm = new OrthoCameraModel();
                    break;
            }
            Debug.Assert(cm != null);

            cm.Name = dc.NameValue;

            m_models.Cameras.Models.Add(cm);
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
