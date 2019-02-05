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
        private RenderTargetSelectionModel m_model;

        public RenderTargetSelectionItem(Core.RenderTarget target, RenderTargetSelectionModel model)
        {
            m_target = target;
            m_model = model;
        }

        public Core.RenderTarget Target { get => m_target;}

        public string TargetName { get => Enum.GetName(typeof(Core.RenderTarget), m_target); }

        public bool Enabled
        {
            get => m_enabled;
            set
            {
                if (value == m_enabled) return;
                m_model.TargetStatus[(int)m_target].Enabled = value;
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
                m_model.TargetStatus[(int)m_target].VarianceEnabled = value;
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
                if(value)
                {
                    m_model.IsVarianceVisible = false;
                    m_model.VisibleTarget = m_target;
                }
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
                if(value)
                {
                    Visible = true;
                    m_model.IsVarianceVisible = true;
                }
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
                var targetItem = new RenderTargetSelectionItem(target, m_models.RenderTargetSelection);
                targetItem.Enabled = m_models.RenderTargetSelection.TargetStatus[(int)target].Enabled;
                targetItem.EnabledVariance = m_models.RenderTargetSelection.TargetStatus[(int)target].VarianceEnabled;
                if (target == m_models.RenderTargetSelection.VisibleTarget)
                {
                    targetItem.Visible = true;
                    targetItem.VarianceVisible = m_models.RenderTargetSelection.IsVarianceVisible;
                }
                TargetData.Add(targetItem);

                m_models.RenderTargetSelection.TargetStatus[(int)target].PropertyChanged += OnModelChanged;
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
                    if (!m_models.Renderer.IsRendering)
                        m_models.Renderer.Iterate(1u);
                    break;
                case nameof(RenderTargetSelectionModel.IsVarianceVisible):
                    TargetData[(int)m_models.RenderTargetSelection.VisibleTarget].VarianceVisible = m_models.RenderTargetSelection.IsVarianceVisible;
                    if(!m_models.Renderer.IsRendering)
                        m_models.Renderer.Iterate(1u);
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
