using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using gui.Annotations;
using gui.Utility;
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

        private Vec3<float> m_position = new Vec3<float>();
        public Vec3<float> Position
        {
            get => m_position;
            set
            {
                if (Equals(m_position, value)) return;
                m_position = value;
                OnPropertyChanged(nameof(Position));
            }
        }

        private Vec3<float> m_viewDir = new Vec3<float>();
        public Vec3<float> ViewDirection
        {
            get => m_viewDir;
            set
            {
                if (Equals(m_viewDir, value)) return;
                m_viewDir = value;
                OnPropertyChanged(nameof(ViewDirection));
            }
        }

        private Vec3<float> m_upDir = new Vec3<float>();
        public Vec3<float> Up
        {
            get => m_upDir;
            set
            {
                if (Equals(m_upDir, value)) return;
                m_upDir = value;
                OnPropertyChanged(nameof(Up));
            }
        }

        private float m_near = 1e-5f;
        public float Near
        {
            get => m_near;
            set
            {
                if (m_near == value) return;
                m_near = value;
                OnPropertyChanged(nameof(Near));
            }
        }

        private float m_far = 1e5f;
        public float Far
        {
            get => m_far;
            set
            {
                if (m_far == value) return;
                m_far = value;
                OnPropertyChanged(nameof(Far));
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
