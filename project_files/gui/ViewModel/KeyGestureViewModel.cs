using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Annotations;
using gui.Command;
using gui.Model;

namespace gui.ViewModel
{
    /// <summary>
    /// Binds the key bindings from the settings model to input bindings from the main window
    /// </summary>
    public class KeyGestureViewModel
    {
        private readonly Models m_models;
        private readonly ICommand m_playPauseCommand;
        private readonly ICommand m_resetCommand;
        private readonly ICommand m_toggleCameraMovementCommand;
        private readonly ICommand m_screenshotCommand;
        private readonly ICommand m_loadSceneCommand;

        public KeyGestureViewModel(Models models)
        {
            m_models = models;

            m_playPauseCommand = new PlayPauseCommand(models);
            m_resetCommand = new ResetCommand(models);
            m_screenshotCommand = new ScreenShotCommand(models);
            m_loadSceneCommand = new LoadSceneCommand(m_models);
            m_toggleCameraMovementCommand = new ActionCommand(() =>
                models.Settings.AllowCameraMovement = !models.Settings.AllowCameraMovement);

            RefreshCommand(models.Settings.PlayPauseGesture, m_playPauseCommand);
            RefreshCommand(models.Settings.ResetGesture, m_resetCommand);
            RefreshCommand(models.Settings.ScreenshotGesture, m_screenshotCommand);
            RefreshCommand(new KeyGesture(Key.O, ModifierKeys.Control), m_loadSceneCommand);
            RefreshCommand(models.Settings.ToggleCameraMovementGesture, m_toggleCameraMovementCommand);

            m_models.Settings.PropertyChanged += SettingsOnPropertyChanged;
        }

        /// <summary>
        /// removes old command key binding and adds the new one (if gesture != null)
        /// </summary>
        /// <param name="gesture">can be null</param>
        /// <param name="command"></param>
        private void RefreshCommand(KeyGesture gesture, ICommand command)
        {
            var inputBindings = m_models.App.Window.InputBindings;
            // remove old command
            for (var i = 0; i < inputBindings.Count; ++i)
            {
                if (!ReferenceEquals(inputBindings[i].Command, command)) continue;

                inputBindings.RemoveAt(i);
                break;
            }
            
            if(gesture == null) return;

            // add new binding
            inputBindings.Add(new KeyBinding(command, gesture));
        }

        private void SettingsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(SettingsModel.PlayPauseGesture):
                    RefreshCommand(m_models.Settings.PlayPauseGesture, m_playPauseCommand);
                    break;
                case nameof(SettingsModel.ResetGesture):
                    RefreshCommand(m_models.Settings.ResetGesture, m_resetCommand);
                    break;
                case nameof(SettingsModel.ScreenshotGesture):
                    RefreshCommand(m_models.Settings.ScreenshotGesture, m_screenshotCommand);
                    break;
                case nameof(SettingsModel.ToggleCameraMovementGesture):
                    RefreshCommand(m_models.Settings.ToggleCameraMovementGesture, m_toggleCameraMovementCommand);
                    break;
            }
        }
    }
}
