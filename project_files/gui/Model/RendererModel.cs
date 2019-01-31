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
        private Core.RendererType m_type = Core.RendererType.CPU_PT;

        public Semaphore RenderLock = new Semaphore(1, 1);

        public RendererModel()
        {
            // Initial state: renderer paused
            RenderLock.WaitOne();
        }

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

        public void UpdateIterationCount()
        {
            OnPropertyChanged(nameof(Iteration));
        }

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

        public static string GetRendererName(Core.RendererType type)
        {
            switch(type)
            {
                case Core.RendererType.CPU_PT: return "Pathtracer (CPU)";
                case Core.RendererType.GPU_PT: return "Pathtracer (GPU)";
            }
            return "Unknown";
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
