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
        private RenderTargetSelectionModel m_model;
        private ICommand m_reset;

        public ObservableCollection<RenderTargetSelectionItem> TargetData { get; set; } =
            new ObservableCollection<RenderTargetSelectionItem>();

        public RenderTargetSelectionViewModel(RenderTargetSelectionModel model, ICommand reset)
        {
            m_model = model;
            m_reset = reset;

            var targets = Enum.GetValues(typeof(Core.RenderTarget));
            foreach(Core.RenderTarget target in targets)
            {
                var targetItem = new RenderTargetSelectionItem(target);
                targetItem.PropertyChanged += OnSelectionChanged;
                targetItem.Enabled = m_model.TargetStatus[(int)target].Enabled;
                targetItem.EnabledVariance = m_model.TargetStatus[(int)target].VarianceEnabled;
                if (target == m_model.VisibleTarget)
                {
                    targetItem.Visible = true;
                    targetItem.VarianceVisible = m_model.IsVarianceVisible;
                }
                TargetData.Add(targetItem);

                m_model.TargetStatus[(int)target].PropertyChanged += OnModelChanged;
            }
            m_model.PropertyChanged += OnModelChanged;
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
                    TargetData[(int)m_model.VisibleTarget].Visible = true;
                    TargetData[(int)m_model.VisibleTarget].VarianceVisible = m_model.IsVarianceVisible;
                    break;
                case nameof(RenderTargetSelectionModel.IsVarianceVisible):
                    TargetData[(int)m_model.VisibleTarget].VarianceVisible = m_model.IsVarianceVisible;
                    break;
                case nameof(RenderTargetSelectionModel.TargetEnabledStatus.Enabled):
                {
                    var target = sender as RenderTargetSelectionModel.TargetEnabledStatus;
                    if (!TargetData[(int)target.Target].Enabled && target.Enabled && m_reset.CanExecute(null))
                        m_reset.Execute(null);
                    TargetData[(int)target.Target].Enabled = target.Enabled;
                }   break;
                case nameof(RenderTargetSelectionModel.TargetEnabledStatus.VarianceEnabled):
                {
                    var target = sender as RenderTargetSelectionModel.TargetEnabledStatus;
                    TargetData[(int)target.Target].EnabledVariance = target.VarianceEnabled;
                    if (!TargetData[(int)target.Target].EnabledVariance && target.VarianceEnabled && m_reset.CanExecute(null))
                        m_reset.Execute(null);
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
                    m_model.TargetStatus[(int)target.Target].Enabled = target.Enabled;
                    break;
                case nameof(RenderTargetSelectionItem.EnabledVariance):
                    m_model.TargetStatus[(int)target.Target].VarianceEnabled = target.EnabledVariance;
                    break;
                case nameof(RenderTargetSelectionItem.Visible):
                    if(target.Visible)
                    {
                        target.Enabled = true;
                        m_model.VisibleTarget = target.Target;
                    }
                    break;
                case nameof(RenderTargetSelectionItem.VarianceVisible):
                    // TODO: we hope that the newly selected radio button is pressed last;
                    // is that guaranteed?
                    if(target.VarianceVisible)
                    {
                        if(m_model.VisibleTarget == target.Target)
                        {
                            m_model.IsVarianceVisible = true;
                        } else
                        {
                            m_model.VisibleTarget = target.Target;
                            target.VarianceVisible = true;
                            m_model.IsVarianceVisible = target.VarianceVisible;
                        }
                    } else
                    {
                        m_model.IsVarianceVisible = false;
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
