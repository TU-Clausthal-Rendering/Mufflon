using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
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

        public bool IsRendering
        {
            get => m_isRendering;
            set
            {
                if(m_isRendering == value) return;
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
            OnPropertyChanged(nameof(Iteration));
        }

        public void reset()
        {
            m_iteration = 0u;
            if (!Core.render_reset())
                throw new Exception(Core.core_get_dll_error());
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
