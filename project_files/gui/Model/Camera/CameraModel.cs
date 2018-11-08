using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using gui.Annotations;
using gui.ViewModel;
using gui.ViewModel.Camera;

namespace gui.Model.Camera
{
    /// <summary>
    /// Base class for camera models
    /// </summary>
    public abstract class CameraModel : INotifyPropertyChanged
    {
        public enum CameraType
        {
            Pinhole,
            Focus,
            Ortho
        }

        public abstract CameraType Type { get; }

        private string m_name = String.Empty;

        public string Name
        {
            get => m_name;
            set
            {
                if (value == null || value == m_name) return;
                m_name = value;
                OnPropertyChanged(nameof(Name));
            }
        }

        private bool m_isSelected = false;
        // indicates if this camera should be used for the renderer
        public bool IsSelected
        {
            get => m_isSelected;
            set
            {
                if (value == m_isSelected) return;
                m_isSelected = value;
                OnPropertyChanged(nameof(IsSelected));
            }
        }

        /// <summary>
        /// creates a new view model based on this model
        /// </summary>
        /// <param name="models"></param>
        /// <returns></returns>
        public abstract CameraViewModel CreateViewModel(Models models);

        #region PropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
}
