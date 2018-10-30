using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using gui.Annotations;

namespace gui.Model
{
    public class CameraModel : INotifyPropertyChanged
    {
        public CameraModel(CameraType type)
        {
            Type = type;
        }

        public enum CameraType
        {
            Pinhole,
            Focus,
            Ortho
        }

        public CameraType Type { get; }

        private float m_fov = 25.0f;

        public float Fov
        {
            get
            {
                Debug.Assert(Type == CameraType.Pinhole);
                return m_fov;
            }
            set
            {
                Debug.Assert(Type == CameraType.Pinhole);
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_fov) return;
                m_fov = value;
                OnPropertyChanged(nameof(Fov));
            }
        }

        private float m_focalLength = 35.0f;

        public float FocalLength
        {
            get
            {
                Debug.Assert(Type == CameraType.Focus);
                return m_focalLength;
            }
            set
            {
                Debug.Assert(Type == CameraType.Focus);
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_focalLength) return;
                m_focalLength = value;
                OnPropertyChanged(nameof(FocalLength));
            }
        }

        private float m_chipHeight = 24.0f;

        public float ChipHeight
        {
            get
            {
                Debug.Assert(Type == CameraType.Focus);
                return m_chipHeight;
            }
            set
            {
                Debug.Assert(Type == CameraType.Focus);
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_chipHeight) return;
                m_chipHeight = value;
                OnPropertyChanged(nameof(ChipHeight));
            }
        }

        private float m_focusDistance;

        public float FocusDistance
        {
            get
            {
                Debug.Assert(Type == CameraType.Focus);
                return m_focusDistance;
            }
            set
            {
                Debug.Assert(Type == CameraType.Focus);
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_focusDistance) return;
                m_focusDistance = value;
                OnPropertyChanged(nameof(FocusDistance));
            }
        }

        private float m_aperture;

        public float Aperture
        {
            get
            {
                Debug.Assert(Type == CameraType.Focus);
                return m_aperture;
            }
            set
            {
                Debug.Assert(Type == CameraType.Focus);
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_aperture) return;
                m_aperture = value;
                OnPropertyChanged(nameof(Aperture));
            }
        }

        private float m_width;

        public float Width
        {
            get
            {
                Debug.Assert(Type == CameraType.Ortho);
                return m_aperture;
            }
            set
            {
                Debug.Assert(Type == CameraType.Ortho);
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_width) return;
                m_width = value;
                OnPropertyChanged(nameof(Width));
            }
        }

        private float m_height;

        public float Height
        {
            get
            {
                Debug.Assert(Type == CameraType.Ortho);
                return m_height;
            }
            set
            {
                Debug.Assert(Type == CameraType.Ortho);
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_height) return;
                m_height = value;
                OnPropertyChanged(nameof(Height));
            }
        }

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
