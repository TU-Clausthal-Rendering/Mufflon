using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using gui.Annotations;
using gui.Properties;

namespace gui.Model
{
    /// <summary>
    /// Information about the current viewport (backbuffer size, viewport size, viewport offset (scrolling))
    /// </summary>
    public class ViewportModel : INotifyPropertyChanged
    {
        // Zoom of the viewport
        private float m_zoom = 1f;
        public float Zoom
        {
            get => m_zoom;
            set
            {
                var clamped = Math.Min(Math.Max(value, 0.01f), 100.0f);
                if (clamped == m_zoom) return;
                m_zoom = clamped;
                OnPropertyChanged(nameof(Zoom));
                OnPropertyChanged(nameof(DesiredWidth));
                OnPropertyChanged(nameof(DesiredHeight));
            }
        }

        private int m_renderWidth = 800;
        // actual width of the backbuffer
        public int RenderWidth
        {
            get => m_renderWidth;
            set
            {
                if (value == m_renderWidth) return;
                m_renderWidth = value;
                OnPropertyChanged(nameof(RenderWidth));
            }
        }

        private int m_renderHeight = 600;
        // actual height of the backbuffer
        public int RenderHeight
        {
            get => m_renderHeight;
            set
            {
                if (value == m_renderHeight) return;
                m_renderHeight = value;
                OnPropertyChanged(nameof(RenderHeight));
            }
        }

        private int m_width = 0;
        // width of the (visible) viewport (ie. size of the border, includes zoom)
        public int Width
        {
            get => m_width;
            set
            {
                if (value == m_width) return;
                m_width = Math.Min(value, DesiredWidth); ;
                OnPropertyChanged(nameof(Width));
            }
        }

        private int m_height = 0;
        // height of the (visible) viewport (ie. size of the border, includes zoom)
        public int Height
        {
            get => m_height;
            set
            {
                if(value == m_height) return;
                m_height = Math.Min(value, DesiredHeight);
                OnPropertyChanged(nameof(Height));
            }
        }

        private int m_offsetX = 0;
        // viewport offset, includes zoom
        public int OffsetX
        {
            get => m_offsetX;
            set
            {
                Debug.Assert(value >= 0);
                Debug.Assert(value < DesiredWidth);
                Debug.Assert(value + Width <= DesiredWidth);
                if (value == m_offsetX) return;
                m_offsetX = value;
                OnPropertyChanged(nameof(OffsetX));
            }
        }

        private int m_offsetY = 0;
        // viewport offset, includes zoom
        public int OffsetY
        {
            get => m_offsetY;
            set
            {
                Debug.Assert(value >= 0);
                Debug.Assert(value < DesiredHeight);
                Debug.Assert(value + Height <= DesiredHeight);
                if (value == m_offsetY) return;
                m_offsetY = value;
                OnPropertyChanged(nameof(OffsetY));
            }
        }

        private bool m_allowMovement = Settings.Default.AllowCameraMovement;
        public bool AllowMovement
        {
            get => m_allowMovement;
            set
            {
                if (value == m_allowMovement) return;
                m_allowMovement = value;
                OnPropertyChanged(nameof(AllowMovement));
            }
        }

        // effective maximum size including zoom
        public int DesiredWidth => (int)(RenderWidth * Zoom);

        // effective maximum size including zoom
        public int DesiredHeight => (int)(RenderHeight * Zoom);

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
