using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Input;
using gui.Annotations;
using gui.Command;
using gui.Dll;
using gui.Model;

namespace gui.ViewModel
{
    public class ToolbarViewModel : INotifyPropertyChanged
    {
        private readonly Models m_models;

        public ToolbarViewModel(Models models)
        {
            m_models = models;

            PlayPauseCommand = new PlayPauseCommand(models);
            ResetCommand = new ResetCommand(models);
            SaveScreenShotCommand = new ScreenShotCommand(models);
            ToggleCameraMovementCommand = new ActionCommand(() =>
                models.Settings.AllowCameraMovement = !models.Settings.AllowCameraMovement);
            EnterFreeFlightMode = new EnterFreeFlightMode(models);
            OneIterationCommand = new PerformIterationsCommand(m_models, 1u);
            NIterationsCommand = new PerformNIterationsCommand(m_models);
            ContinuousSequenceRenderCommand = new RenderSequenceCommand(m_models, -1, -1, -1, false);

            m_models.Renderer.PropertyChanged += RendererOnPropertyChanged;
            m_models.Settings.PropertyChanged += SettingsOnPropertyChanged;
        }

        private void SettingsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(SettingsModel.AllowCameraMovement):
                    OnPropertyChanged(nameof(CameraMoveIconVisibility));
                    break;
            }
        }

        private void RendererOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(RendererModel.IsRendering):
                    OnPropertyChanged(nameof(PauseIconVisibility));
                    OnPropertyChanged(nameof(PlayIconVisibility));
                    break;
            }
        }

        public uint? Iterations
        {
            get => m_models.Toolbar.Iterations;
            set
            {
                m_models.Toolbar.Iterations = value;
                if (m_models.Toolbar.Iterations.HasValue)
                    m_models.Settings.LastNIterationCommand = m_models.Toolbar.Iterations.Value;
            }
        }

        public Visibility PlayIconVisibility =>
            m_models.Renderer.IsRendering ? Visibility.Collapsed : Visibility.Visible;

        public Visibility PauseIconVisibility =>
            m_models.Renderer.IsRendering ? Visibility.Visible : Visibility.Collapsed;

        public Visibility CameraMoveIconVisibility =>
            m_models.Settings.AllowCameraMovement ? Visibility.Collapsed : Visibility.Visible;

        public ICommand PlayPauseCommand { get; }
        public ICommand ResetCommand { get; }
        public ICommand SaveScreenShotCommand { get; }
        public ICommand ToggleCameraMovementCommand { get; }
        public ICommand EnterFreeFlightMode { get; }
        public ICommand OneIterationCommand { get; }
        public ICommand NIterationsCommand { get; }
        public ICommand ContinuousSequenceRenderCommand { get; }
        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
