﻿using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Input;
using gui.Annotations;
using gui.Command;
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
            ResetCommand = new ResetCommand(models, PlayPauseCommand);
            SaveScreenShotCommand = new ScreenShotCommand(models);
            ToggleCameraMovementCommand = new ToggleCameraMovementCommand(models);

            m_models.Renderer.PropertyChanged += RendererOnPropertyChanged;
            m_models.Viewport.PropertyChanged += ViewportOnPropertyChanged;
        }

        private void ViewportOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            if (sender == m_models.Viewport && args.PropertyName == nameof(Models.Viewport.AllowMovement))
                OnPropertyChanged(nameof(CameraMoveIconVisibility));
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

        public Visibility PlayIconVisibility =>
            m_models.Renderer.IsRendering ? Visibility.Collapsed : Visibility.Visible;

        public Visibility PauseIconVisibility =>
            m_models.Renderer.IsRendering ? Visibility.Visible : Visibility.Collapsed;

        public Visibility CameraMoveIconVisibility =>
            m_models.Viewport.AllowMovement ? Visibility.Collapsed : Visibility.Visible;

        public IGesturedCommand PlayPauseCommand { get; }
        public IGesturedCommand ResetCommand { get; }
        public IGesturedCommand SaveScreenShotCommand { get; }
        public IGesturedCommand ToggleCameraMovementCommand { get; }
        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
