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
                if (value)
                    RenderLock.Release();
                else
                    RenderLock.WaitOne();
                m_isRendering = value;
                OnPropertyChanged(nameof(IsRendering));
            }
        }

        public uint Iteration {
            get => m_iteration;
            set
            {
                if (m_iteration == value) return;
                m_iteration = value;
                OnPropertyChanged(nameof(Iteration));
            }
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

        public static string getRendererName(Core.RendererType type)
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
