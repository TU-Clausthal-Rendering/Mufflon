using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Model;

namespace gui.Command
{
    class PerformIterationsCommand : ICommand
    {
        private readonly Models m_models;
        private readonly uint m_iterations;

        public PerformIterationsCommand(Models models, uint iterations)
        {
            m_models = models;
            m_iterations = iterations;
            m_models.PropertyChanged += ModelsOnPropertyChanged;
            m_models.Renderer.PropertyChanged += RendererOnPropertyChanged;
        }

        private void ModelsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            if(args.PropertyName == nameof(Models.World))
                OnCanExecuteChanged();
        }

        private void RendererOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            if (sender == m_models.Renderer && args.PropertyName == nameof(Models.Renderer.IsRendering))
                OnCanExecuteChanged();
        }

        public bool CanExecute(object parameter)
        {
            return m_models.World != null && !m_models.Renderer.IsRendering;
        }

        public void Execute(object parameter)
        {
            m_models.Renderer.Iterate(m_iterations);
        }

        public event EventHandler CanExecuteChanged;

        protected virtual void OnCanExecuteChanged()
        {
            CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        }
    }
}
