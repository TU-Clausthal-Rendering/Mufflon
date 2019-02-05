using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using gui.Annotations;
using gui.Dll;

namespace gui.Model
{
    public class RenderTargetSelectionModel : INotifyPropertyChanged
    {
        public class TargetEnabledStatus : INotifyPropertyChanged
        {
            private Core.RenderTarget m_target;

            public TargetEnabledStatus(Core.RenderTarget target)
            {
                m_target = target;
            }

            public Core.RenderTarget Target { get => m_target; }

            public bool Enabled
            {
                get => Core.render_is_render_target_enabled(Target, false);
                set
                {
                    if (value == Core.render_is_render_target_enabled(Target, false)) return;
                    if (value)
                    {
                        if (!Core.render_enable_render_target(m_target, VarianceEnabled ? 1u : 0u))
                            throw new Exception(Core.core_get_dll_error());
                    }
                    else
                    {
                        if (!Core.render_disable_render_target(m_target, 0u))
                            throw new Exception(Core.core_get_dll_error());
                        VarianceEnabled = false;
                    }
                    OnPropertyChanged(nameof(Enabled));
                }
            }

            public bool VarianceEnabled
            {
                get => Core.render_is_render_target_enabled(Target, true);
                set
                {
                    if (value == Core.render_is_render_target_enabled(Target, true)) return;
                    if(value)
                    {
                        Enabled = true;
                    } else
                    {
                        if (!Core.render_disable_render_target(m_target, 1u))
                            throw new Exception(Core.core_get_dll_error());
                    }
                    OnPropertyChanged(nameof(VarianceEnabled));
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

        private List<TargetEnabledStatus> m_targetStatus = new List<TargetEnabledStatus>();
        private Core.RenderTarget m_visibleTarget = Core.RenderTarget.Radiance;
        private bool m_isVarianceVisible = false;

        public RenderTargetSelectionModel()
        {
            var targets = Enum.GetValues(typeof(Core.RenderTarget));
            foreach(Core.RenderTarget target in targets)
                m_targetStatus.Add(new TargetEnabledStatus(target));
            // Default: radiance is enabled
            m_targetStatus[(int)Core.RenderTarget.Radiance].Enabled = true;
        }

        public IReadOnlyList<TargetEnabledStatus> TargetStatus { get => m_targetStatus; }

        public Core.RenderTarget VisibleTarget
        {
            get => m_visibleTarget;
            set
            {
                if (value == m_visibleTarget) return;
                TargetStatus[(int)value].Enabled = true;
                m_visibleTarget = value;
                OnPropertyChanged(nameof(VisibleTarget));
            }
        }

        public bool IsVarianceVisible
        {
            get => m_isVarianceVisible;
            set
            {
                if (value == m_isVarianceVisible) return;
                if(value)
                    TargetStatus[(int)VisibleTarget].VarianceEnabled = true;
                m_isVarianceVisible = value;
                OnPropertyChanged(nameof(IsVarianceVisible));
            }
        }
        
        public static string getRenderTargetName(Core.RenderTarget target, bool variance)
        {
            string val = Enum.GetName(typeof(Core.RenderTarget), target);
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
