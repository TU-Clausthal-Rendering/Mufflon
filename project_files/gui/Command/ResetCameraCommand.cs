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
        private readonly Models m_models;

        private CameraModel m_camera;

        public ResetCameraCommand(Models models, CameraModel camera)
        {
            m_models = models;
            m_camera = camera;

            models.Settings.PropertyChanged += SettingsPropertyChanged;
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
            return m_models.Settings.AllowCameraMovement;
        }

        public virtual void Execute(object parameter)
        {
            m_camera.Reset(m_models.World.AnimationFrameCurrent);
        }

        public event EventHandler CanExecuteChanged;

        protected virtual void OnCanExecuteChanged()
        {
            CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        }
    }
}
