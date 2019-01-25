﻿using System;
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
        private Core.RendererType m_type = Core.RendererType.CPU_PT;
        private Core.RenderTarget m_target = Core.RenderTarget.RADIANCE;
        private bool m_varianceTarget = false;

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

        public uint Iteration {
            get => Core.render_get_current_iteration();
        }

        public void updateIterationCount()
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

        // The render target that will be displayed
        public Core.RenderTarget RenderTarget
        {
            get => m_target;
            set
            {
                if (m_target == value) return;
                m_target = value;
                OnPropertyChanged(nameof(RenderTarget));
            }
        }

        public bool RenderTargetVariance
        {
            get => m_varianceTarget;
            set
            {
                if (m_varianceTarget == value) return;
                m_varianceTarget = value;
                OnPropertyChanged(nameof(RenderTargetVariance));
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

        public static string getRenderTargetName(Core.RenderTarget target, bool variance)
        {
            string val;
            switch(target)
            {
                case Core.RenderTarget.ALBEDO: val = "Albedo"; break;
                case Core.RenderTarget.LIGHTNESS: val = "Lightness"; break;
                case Core.RenderTarget.NORMAL: val = "Normal"; break;
                case Core.RenderTarget.POSITION: val = "Position"; break;
                case Core.RenderTarget.RADIANCE: val = "Radiance"; break;
                default: val = "Unknown"; break;
            }
            if (variance)
                val += " (Variance)";
            return val;
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
