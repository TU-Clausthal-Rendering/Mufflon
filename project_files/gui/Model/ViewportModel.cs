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
    /// <summary>
    /// Information about the current viewport (backbuffer size, viewport size, viewport offset (scrolling))
    /// </summary>
    public class ViewportModel : INotifyPropertyChanged
    {
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
        // width of the (visible) viewport
        public int Width
        {
            get => m_width;
            set
            {
                if (value == m_width) return;
                m_width = value;
                OnPropertyChanged(nameof(Width));
            }
        }

        private int m_height = 0;
        // height of the (visible) viewport
        public int Height
        {
            get => m_height;
            set
            {
                if(value == m_height) return;
                m_height = value;
                OnPropertyChanged(nameof(Height));
            }
        }

        private int m_offsetX = 0;
        // viewport offset
        public int OffsetX
        {
            get => m_offsetX;
            set
            {
                Debug.Assert(value >= 0);
                Debug.Assert(value < RenderWidth);
                Debug.Assert(value + Width <= RenderWidth);
                if (value == m_offsetX) return;
                m_offsetX = value;
                OnPropertyChanged(nameof(OffsetX));
            }
        }

        private int m_offsetY = 0;
        // viewport offset
        public int OffsetY
        {
            get => m_offsetY;
            set
            {
                Debug.Assert(value >= 0);
                Debug.Assert(value < RenderHeight);
                Debug.Assert(value + Height <= RenderHeight);
                if (value == m_offsetY) return;
                m_offsetY = value;
                OnPropertyChanged(nameof(OffsetY));
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
