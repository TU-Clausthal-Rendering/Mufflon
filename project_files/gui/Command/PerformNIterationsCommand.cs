using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Dll;
using gui.Model;
using gui.View;

namespace gui.Command
{
    class PerformNIterationsCommand : ICommand
    {
        private readonly Models m_models;

        public PerformNIterationsCommand(Models models)
        {
            m_models = models;
            m_models.PropertyChanged += ModelsOnPropertyChanged;
            m_models.Renderer.PropertyChanged += RendererOnPropertyChanged;
            m_models.Toolbar.PropertyChanged += IterationsOnPropertyChanged;
        }

        private void ModelsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            if (args.PropertyName == nameof(Models.World))
                OnCanExecuteChanged();
        }

        private void RendererOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            if (sender == m_models.Renderer && args.PropertyName == nameof(Models.Renderer.IsRendering))
                OnCanExecuteChanged();
        }

        private void IterationsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            if (sender == m_models.Toolbar && args.PropertyName == nameof(Models.Toolbar.Iterations))
                OnCanExecuteChanged();
        }

        public bool CanExecute(object parameter)
        {
            return m_models.World != null && !m_models.Renderer.IsRendering
                && m_models.Toolbar.Iterations.HasValue;
        }

        public void Execute(object parameter)
        {
            m_models.Renderer.Iterate(m_models.Toolbar.Iterations.Value);
        }

        public event EventHandler CanExecuteChanged;

        protected virtual void OnCanExecuteChanged()
        {
            CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        }
    }
}
