using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using gui.Annotations;
using gui.Dll;

namespace gui.Model
{
    public class RenderTarget : INotifyPropertyChanged
    {
        public RenderTarget(UInt32 targetIndex)
        {
            TargetIndex = targetIndex;
            Name = Core.render_get_render_target_name(TargetIndex);
        }

        public UInt32 TargetIndex { get; private set; }

        public string Name { get; private set; }

        public bool Enabled
        {
            get => Core.render_is_render_target_enabled(TargetIndex, false);
            set
            {
                if (value == Core.render_is_render_target_enabled(TargetIndex, false)) return;
                if (value)
                {
                    if (!Core.render_enable_render_target(TargetIndex, VarianceEnabled ? 1u : 0u))
                        throw new Exception(Core.core_get_dll_error());
                }
                else
                {
                    VarianceEnabled = false;
                    if (!Core.render_disable_render_target(TargetIndex, 0u))
                        throw new Exception(Core.core_get_dll_error());
                }
                OnPropertyChanged(nameof(Enabled));
            }
        }

        public bool VarianceEnabled
        {
            get => Core.render_is_render_target_enabled(TargetIndex, true);
            set
            {
                if (value == Core.render_is_render_target_enabled(TargetIndex, true)) return;
                if (value)
                {
                    Enabled = true;
                    if (!Core.render_enable_render_target(TargetIndex, 1u))
                        throw new Exception(Core.core_get_dll_error());
                }
                else
                {
                    if (!Core.render_disable_render_target(TargetIndex, 1u))
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

    public class RenderTargetSelectionModel : INotifyPropertyChanged
    {

        private List<RenderTarget> m_targetStatus = new List<RenderTarget>();
        private RenderTarget m_visibleTarget;
        private bool m_isVarianceVisible = false;

        public RenderTargetSelectionModel()
        {
            UInt32 targetCount = Core.render_get_render_target_count();
            for(UInt32 i = 0u; i < targetCount; ++i)
                m_targetStatus.Add(new RenderTarget(i));
            VisibleTarget = Targets[0];
        }

        public IReadOnlyList<RenderTarget> Targets { get => m_targetStatus; }

        public RenderTarget VisibleTarget
        {
            get => m_visibleTarget;
            set
            {
                if (value == m_visibleTarget) return;
                m_visibleTarget = value;
                m_visibleTarget.Enabled = true;
                OnPropertyChanged(nameof(VisibleTarget));
            }
        }

        public bool IsVarianceVisible
        {
            get => m_isVarianceVisible;
            set
            {
                if (value == m_isVarianceVisible) return;
                if (value)
                {
                    VisibleTarget.Enabled = true;
                    VisibleTarget.VarianceEnabled = true;
                }

                m_isVarianceVisible = value;
                OnPropertyChanged(nameof(IsVarianceVisible));
            }
        }
        
        public static string getRenderTargetName(UInt32 index, bool variance)
        {
            string name = Core.render_get_render_target_name(index);
            if (variance)
                name += " (Variance)";
            return name;
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
