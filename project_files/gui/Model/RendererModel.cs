using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using gui.Annotations;
using gui.Dll;

namespace gui.Model
{
    /// <summary>
    /// information about the active renderer
    /// </summary>
    public class RendererModel : INotifyPropertyChanged
    {
        private bool m_isRendering = false;
        private UInt32 m_rendererIndex = 0u;
        private UInt32 m_rendererCount = 0u;

        public RendererModel()
        {
            // Initial state: renderer paused
            RenderLock.WaitOne();
        }

        public Semaphore RenderLock = new Semaphore(1, 1);

        public bool IsRendering
        {
            get => m_isRendering;
            set
            {
                if(m_isRendering == value) return;
                m_isRendering = value;
                if (value)
                    RenderLock.Release();
                else
                    RenderLock.WaitOne();
                OnPropertyChanged(nameof(IsRendering));
            }
        }

        public uint Iteration => Core.render_get_current_iteration();

        public void Reset()
        {
            if (!Core.render_reset())
                throw new Exception(Core.core_get_dll_error());
           UpdateIterationCount();
        }

        public void Iterate(uint times)
        {
            for(uint i = 0u; i < times; ++i)
            {
                RenderLock.Release();
                RenderLock.WaitOne();
            }
        }

        public void UpdateIterationCount()
        {
            OnPropertyChanged(nameof(Iteration));
        }

        public UInt32 RendererCount
        {
            get => m_rendererCount;
            set
            {
                if (m_rendererCount == value) return;
                m_rendererCount = value;
                OnPropertyChanged(nameof(RendererCount));
            }
        }

        public UInt32 RendererIndex
        {
            get => m_rendererIndex;
            set
            {
                if (m_rendererIndex == value) return;
                m_rendererIndex = value;
                OnPropertyChanged(nameof(RendererIndex));
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
