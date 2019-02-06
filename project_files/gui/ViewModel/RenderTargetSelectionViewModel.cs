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
        private RenderTargetSelectionModel m_model;
        private RenderTarget m_target;
        private bool m_enabled = false;
        private bool m_varianceEnabled = false;
        private bool m_visible = false;
        private bool m_varianceVisible = false;

        public RenderTargetSelectionItem(RenderTarget target, RenderTargetSelectionModel model)
        {
            m_model = model;
            m_target = target;
            Enabled = m_target.Enabled;
            VarianceEnabled = m_target.VarianceEnabled;
            if (m_target == m_model.VisibleTarget)
            {
                Visible = true;
                if (m_model.IsVarianceVisible)
                    VarianceVisible = true;
            }
        }

        public UInt32 TargetIndex { get => m_target.TargetIndex; }
    
        public string TargetName { get => m_target.Name; }

        public bool Enabled
        {
            get => m_enabled;
            set
            {
                if (value == m_enabled) return;
                m_enabled = value;
                m_target.Enabled = value;
                OnPropertyChanged(nameof(Enabled));
            }
        }

        public bool VarianceEnabled
        {
            get => m_varianceEnabled;
            set
            {
                if (value == m_varianceEnabled) return;
                m_varianceEnabled = value;
                m_target.VarianceEnabled = value;
                OnPropertyChanged(nameof(VarianceEnabled));
            }
        }

        public bool Visible
        {
            get => m_visible;
            set
            {
                if (value == m_visible) return;
                m_visible = value;
                if (value)
                {
                    m_model.IsVarianceVisible = false;
                    m_model.VisibleTarget = m_target;
                }
                OnPropertyChanged(nameof(Visible));
            }
        }

        public bool VarianceVisible
        {
            get => m_varianceVisible;
            set
            {
                if (value == m_varianceVisible) return;
                if (value)
                {
                    Visible = true;
                    m_varianceVisible = true;
                    m_model.IsVarianceVisible = true;
                } else
                {
                    m_varianceVisible = false;
                }
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

            foreach(RenderTarget target in m_models.RenderTargetSelection.Targets)
            {
                TargetData.Add(new RenderTargetSelectionItem(target, m_models.RenderTargetSelection));
                m_models.RenderTargetSelection.Targets[(int)target.TargetIndex].PropertyChanged += OnModelChanged;
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
                    TargetData[(int)m_models.RenderTargetSelection.VisibleTarget.TargetIndex].Visible = true;
                    TargetData[(int)m_models.RenderTargetSelection.VisibleTarget.TargetIndex].VarianceVisible = m_models.RenderTargetSelection.IsVarianceVisible;
                    if (!m_models.Renderer.IsRendering)
                        m_models.Renderer.Iterate(1u);
                    break;
                case nameof(RenderTargetSelectionModel.IsVarianceVisible):
                    TargetData[(int)m_models.RenderTargetSelection.VisibleTarget.TargetIndex].VarianceVisible = m_models.RenderTargetSelection.IsVarianceVisible;
                    if(!m_models.Renderer.IsRendering)
                        m_models.Renderer.Iterate(1u);
                    break;
                case nameof(RenderTarget.Enabled):
                {
                    var target = sender as RenderTarget;
                    TargetData[(int)target.TargetIndex].Enabled = target.Enabled;
                }   break;
                case nameof(RenderTarget.VarianceEnabled):
                {
                    var target = sender as RenderTarget;
                    TargetData[(int)target.TargetIndex].VarianceEnabled = target.VarianceEnabled;
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
