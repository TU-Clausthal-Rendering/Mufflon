using System;
using System.IO;
using System.Windows.Forms;
using System.Windows.Input;
using System.ComponentModel;
using gui.Model.Camera;
using gui.Model;

namespace gui.Command
{
    public class ResetCameraCommand : ICommand
    {
        private readonly SettingsModel m_settings;
        private CameraModel m_camera;

        public ResetCameraCommand(SettingsModel settings, CameraModel camera)
        {
            m_settings = settings;
            m_camera = camera;

            m_settings.PropertyChanged += SettingsPropertyChanged;
        }

        private void SettingsPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(SettingsModel.AllowCameraMovement):
                    OnCanExecuteChanged();
                    break;
            }
        }

        public bool CanExecute(object parameter)
        {
            return m_settings.AllowCameraMovement;
        }

        public virtual void Execute(object parameter)
        {
            m_camera.ResetTransRot();
        }

        public event EventHandler CanExecuteChanged;

        protected virtual void OnCanExecuteChanged()
        {
            CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        }
    }
}
