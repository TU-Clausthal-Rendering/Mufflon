using System;
using System.Collections.Generic;
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
        private uint m_iteration = 0u;
        private Core.RendererType m_type;
        private ManualResetEvent m_iterationComplete = new ManualResetEvent(false);

        public bool IsRendering
        {
            get => m_isRendering;
            set
            {
                if(m_isRendering == value) return;
                if (m_isRendering)
                    waitForCompletedIteration(TimeSpan.FromSeconds(10));
                m_isRendering = value;
                OnPropertyChanged(nameof(IsRendering));
            }
        }

        public uint Iteration { get => m_iteration; }

        public Core.RendererType Type
        {
            get => m_type;
            set
            {
                if (m_type == value) return;
                m_type = value;
                OnPropertyChanged(nameof(Type));
            }
        }

        public void performedIteration()
        {
            ++m_iteration;
            m_iterationComplete.Set();
            OnPropertyChanged(nameof(Iteration));
        }

        public void waitForCompletedIteration()
        {
            m_iterationComplete.Reset();
            m_iterationComplete.WaitOne();
        }

        public void waitForCompletedIteration(TimeSpan timeout)
        {
            m_iterationComplete.Reset();
            m_iterationComplete.WaitOne(Convert.ToInt32(timeout.TotalMilliseconds));
        }

        public void reset()
        {
            if (IsRendering)
            {
                // Briefly pause the rendering process
                IsRendering = false;
                m_iteration = 0u;
                if (!Core.render_reset())
                    throw new Exception(Core.core_get_dll_error());
                // Resume rendering
                IsRendering = true;
            } else
            {
                m_iteration = 0u;
                if (!Core.render_reset())
                    throw new Exception(Core.core_get_dll_error());
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
