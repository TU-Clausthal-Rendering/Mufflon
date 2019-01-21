using System;
using System.IO;
using System.Windows;
using System.Windows.Input;
using gui.Dll;
using gui.Model;
using gui.Properties;

namespace gui.Command
{
    public class ToggleCameraMovementCommand : ICommand
    {
        private readonly Models m_models;

        public ToggleCameraMovementCommand(Models models)
        {
            m_models = models;
        }

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public void Execute(object parameter)
        {
            m_models.Settings.AllowCameraMovement = !m_models.Settings.AllowCameraMovement;
        }

        public event EventHandler CanExecuteChanged
        {
            add {}
            remove {}
        }
    }
}
