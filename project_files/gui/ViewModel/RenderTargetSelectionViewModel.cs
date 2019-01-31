using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Annotations;
using gui.Dll;
using gui.Model;

namespace gui.ViewModel
{
    public class RenderTargetSelectionItem : INotifyPropertyChanged
    {
        private Core.RenderTarget m_target;
        private bool m_enabled = false;
        private bool m_enabledVariance = false;
        private bool m_visible = false;
        private bool m_varianceVisible = false;

        public RenderTargetSelectionItem(Core.RenderTarget target)
        {
            m_target = target;
        }

        public Core.RenderTarget Target { get => m_target;}

        public string TargetName { get => Enum.GetName(typeof(Core.RenderTarget), m_target); }

        public bool Enabled
        {
            get => m_enabled;
            set
            {
                if (value == m_enabled) return;
                m_enabled = value;
                OnPropertyChanged(nameof(Enabled));
            }
        }

        public bool EnabledVariance
        {
            get => m_enabledVariance;
            set
            {
                if (value == m_enabledVariance) return;
                m_enabledVariance = value;
                OnPropertyChanged(nameof(EnabledVariance));
            }
        }

        public bool Visible
        {
            get => m_visible;
            set
            {
                if (value == m_visible) return;
                m_visible = value;
                OnPropertyChanged(nameof(Visible));
            }
        }

        public bool VarianceVisible
        {
            get => m_varianceVisible;
            set
            {
                if (value == m_varianceVisible) return;
                m_varianceVisible = value;
                OnPropertyChanged(nameof(VarianceVisible));
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
    };

    public class RenderTargetSelectionViewModel : INotifyPropertyChanged
    {
        private Models m_models;
        private ICommand m_reset;

        public ObservableCollection<RenderTargetSelectionItem> TargetData { get; set; } =
            new ObservableCollection<RenderTargetSelectionItem>();

        public RenderTargetSelectionViewModel(Models models, ICommand reset)
        {
            m_models = models;
            m_reset = reset;

            var targets = Enum.GetValues(typeof(Core.RenderTarget));
            foreach(Core.RenderTarget target in targets)
            {
                var targetItem = new RenderTargetSelectionItem(target);
                targetItem.Enabled = m_models.RenderTargetSelection.TargetStatus[(int)target].Enabled;
                targetItem.EnabledVariance = m_models.RenderTargetSelection.TargetStatus[(int)target].VarianceEnabled;
                if (target == m_models.RenderTargetSelection.VisibleTarget)
                {
                    targetItem.Visible = true;
                    targetItem.VarianceVisible = m_models.RenderTargetSelection.IsVarianceVisible;
                }
                TargetData.Add(targetItem);

                m_models.RenderTargetSelection.TargetStatus[(int)target].PropertyChanged += OnModelChanged;
                targetItem.PropertyChanged += OnSelectionChanged;
            }
            m_models.RenderTargetSelection.PropertyChanged += OnModelChanged;
            OnPropertyChanged(nameof(TargetData));
        }

        private void OnModelChanged(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(RenderTargetSelectionModel.VisibleTarget):
                    foreach (var data in TargetData)
                    {
                        data.VarianceVisible = false;
                        data.Visible = false;
                    }
                    TargetData[(int)m_models.RenderTargetSelection.VisibleTarget].Visible = true;
                    TargetData[(int)m_models.RenderTargetSelection.VisibleTarget].VarianceVisible = m_models.RenderTargetSelection.IsVarianceVisible;
                    break;
                case nameof(RenderTargetSelectionModel.IsVarianceVisible):
                    TargetData[(int)m_models.RenderTargetSelection.VisibleTarget].VarianceVisible = m_models.RenderTargetSelection.IsVarianceVisible;
                    break;
                case nameof(RenderTargetSelectionModel.TargetEnabledStatus.Enabled):
                {
                    var target = sender as RenderTargetSelectionModel.TargetEnabledStatus;
                    TargetData[(int)target.Target].Enabled = target.Enabled;
                }   break;
                case nameof(RenderTargetSelectionModel.TargetEnabledStatus.VarianceEnabled):
                {
                    var target = sender as RenderTargetSelectionModel.TargetEnabledStatus;
                    TargetData[(int)target.Target].EnabledVariance = target.VarianceEnabled;
                }   break;
            }
        }

        // TODO: disable disabling if target is visible
        private void OnSelectionChanged(object sender, PropertyChangedEventArgs args)
        {
            RenderTargetSelectionItem target = sender as RenderTargetSelectionItem;
            switch (args.PropertyName)
            {
                case nameof(RenderTargetSelectionItem.Enabled):
                {
                    bool wasRunning = m_models.Renderer.IsRendering;
                    m_models.Renderer.IsRendering = false;
                    if (target.Enabled && m_reset.CanExecute(null))
                        m_reset.Execute(null);
                    m_models.RenderTargetSelection.TargetStatus[(int)target.Target].Enabled = target.Enabled;
                    m_models.Renderer.IsRendering = wasRunning;
                }   break;
                case nameof(RenderTargetSelectionItem.EnabledVariance):
                {
                    bool wasRunning = m_models.Renderer.IsRendering;
                    m_models.Renderer.IsRendering = false;
                    if (target.EnabledVariance && m_reset.CanExecute(null))
                        m_reset.Execute(null);
                    m_models.RenderTargetSelection.TargetStatus[(int)target.Target].VarianceEnabled = target.EnabledVariance;
                    m_models.Renderer.IsRendering = wasRunning;
                }   break;
                case nameof(RenderTargetSelectionItem.Visible):
                    if(target.Visible)
                    {
                        target.Enabled = true;
                        m_models.RenderTargetSelection.VisibleTarget = target.Target;
                    }
                    break;
                case nameof(RenderTargetSelectionItem.VarianceVisible):
                    // TODO: we hope that the newly selected radio button is pressed last;
                    // is that guaranteed?
                    if(target.VarianceVisible)
                    {
                        if(m_models.RenderTargetSelection.VisibleTarget == target.Target)
                        {
                            m_models.RenderTargetSelection.IsVarianceVisible = true;
                        } else
                        {
                            m_models.RenderTargetSelection.VisibleTarget = target.Target;
                            target.VarianceVisible = true;
                            m_models.RenderTargetSelection.IsVarianceVisible = target.VarianceVisible;
                        }
                    } else
                    {
                        m_models.RenderTargetSelection.IsVarianceVisible = false;
                    }
                    break;
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
