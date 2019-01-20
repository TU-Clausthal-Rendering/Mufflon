using System;
using System.IO;
using System.Windows;
using System.Windows.Input;
using gui.Dll;
using gui.Model;
using gui.Properties;

namespace gui.Command
{
    public class ToggleCameraMovementCommand : IGesturedCommand
    {
        private Models m_models;

        public ToggleCameraMovementCommand(Models models) : base("ToggleCameraMovementGesture")
        {
            m_models = models;
        }

        public override bool CanExecute(object parameter)
        {
            return true;
        }

        public override void Execute(object parameter)
        {
            m_models.Settings.AllowCameraMovement = !m_models.Settings.AllowCameraMovement;
        }

        public override event EventHandler CanExecuteChanged
        {
            add
            {
            }
            remove
            {
            }
        }
    }
}
