using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Model;
using gui.Model.Camera;

namespace gui.Command
{
    public class RemoveCameraCommand : ICommand
    {
        private readonly Models m_models;
        private readonly CameraModel m_model;

        public RemoveCameraCommand(Models models, CameraModel model)
        {
            m_models = models;
            m_model = model;
        }

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public void Execute(object parameter)
        {
            m_models.Cameras.Models.Remove(m_model);
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
