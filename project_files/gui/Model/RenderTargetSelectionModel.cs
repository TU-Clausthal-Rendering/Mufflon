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
            LastEnabledStatus = Enabled;
            LastVarianceEnabledStatus = VarianceEnabled;
        }

        // Used to query enabled status after a renderer switch
        public bool LastEnabledStatus { get; private set; }
        public bool LastVarianceEnabledStatus { get; private set; }

        public UInt32 TargetIndex { get; private set; }

        public string Name { get; private set; }

        public bool Enabled
        {
            get => Core.render_is_render_target_enabled(Name, false);
            set
            {
                LastEnabledStatus = value;
                if (value == Core.render_is_render_target_enabled(Name, false)) return;
                if (value)
                {
                    if (!Core.render_enable_render_target(Name, VarianceEnabled))
                        throw new Exception(Core.core_get_dll_error());
                }
                else
                {
                    VarianceEnabled = false;
                    if (!Core.render_disable_render_target(Name, false))
                        throw new Exception(Core.core_get_dll_error());
                }
                OnPropertyChanged(nameof(Enabled));
            }
        }

        public bool VarianceEnabled
        {
            get => Core.render_is_render_target_enabled(Name, true);
            set
            {
                LastVarianceEnabledStatus = value;
                if (value == Core.render_is_render_target_enabled(Name, true)) return;
                if (value)
                {
                    Enabled = true;
                    if (!Core.render_enable_render_target(Name, true))
                        throw new Exception(Core.core_get_dll_error());
                }
                else
                {
                    if (!Core.render_disable_render_target(Name, true))
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
            UpdateTargetList();
        }

        public void UpdateTargetList()
        {
            var newTargets = new List<RenderTarget>();
            UInt32 targetCount = Core.render_get_render_target_count();
            for (UInt32 i = 0u; i < targetCount; ++i)
            {
                var target = new RenderTarget(i);
                newTargets.Add(target);
                // Try to match enabled/disabled for equal names
                var sameTargets = m_targetStatus.FindAll(x => x.Name == target.Name);
                if(sameTargets.Count != 0)
                {
                    // Only use the first, equal names for same renderer are impossible anyway
                    target.Enabled = sameTargets[0].LastEnabledStatus;
                    target.VarianceEnabled = sameTargets[0].LastVarianceEnabledStatus;
                }
            }
            m_targetStatus = newTargets;
            OnPropertyChanged(nameof(Targets));
            if (targetCount > 0)
            {
                if(VisibleTarget != null)
                {
                    // Try to find a target with the same name as the previous target
                    var sameTargets = m_targetStatus.FindAll(x => x.Name == VisibleTarget.Name);
                    if (sameTargets.Count != 0)
                    {
                        // Only use the first, equal names for same renderer are impossible anyway
                        VisibleTarget = sameTargets[0];
                    }
                    else
                    {
                        VisibleTarget = Targets[0];
                    }
                } else
                {
                    VisibleTarget = Targets[0];
                }
            }
            else
            {
                VisibleTarget = null;
            }
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
