using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using gui.Annotations;
using gui.Model;

namespace gui.ViewModel.Settings
{
    public class KeybindingsViewModel : INotifyPropertyChanged
    {
        private readonly Models m_models;

        public KeybindingsViewModel(Models models)
        {
            m_models = models;
        }

        internal void LoadFromSettings()
        {
            PlayPauseGesture = m_models.Settings.PlayPauseGestureString;
            ResetGesture = m_models.Settings.ResetGestureString;
            ScreenshotGesture = m_models.Settings.ScreenshotGestureString;
            CameraMoveToggleGesture = m_models.Settings.ToggleCameraMovementGestureString;
        }

        internal void StoreSettings()
        {
            var s = m_models.Settings;
            s.PlayPauseGestureString = PlayPauseGesture;
            s.ResetGestureString = ResetGesture;
            s.ScreenshotGestureString = ScreenshotGesture;
            s.ToggleCameraMovementGestureString = CameraMoveToggleGesture;
        }


        public string PlayPauseGesture { get; set; }
        public string ResetGesture { get; set; }
        public string ScreenshotGesture { get; set; }
        public string CameraMoveToggleGesture { get; set; }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
