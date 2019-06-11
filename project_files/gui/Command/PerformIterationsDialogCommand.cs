using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Model;
using gui.View;

namespace gui.Command
{
    class PerformIterationsDialogCommand : ICommand
    {
        private readonly Models m_models;

        public PerformIterationsDialogCommand(Models models)
        {
            m_models = models;
            m_models.PropertyChanged += ModelsOnPropertyChanged;
            m_models.Renderer.PropertyChanged += RendererOnPropertyChanged;
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

        public bool CanExecute(object parameter)
        {
            return m_models.World != null && !m_models.Renderer.IsRendering;
        }

        public void Execute(object parameter)
        {
            IterateNDialog dialog = new IterateNDialog(m_models.Settings.LastNIterationCommand);
            if(dialog.ShowDialog() == true)
            {
                uint newIterCount = dialog.Iterations;
                m_models.Settings.LastNIterationCommand = newIterCount;
                m_models.Renderer.Iterate(newIterCount);
            }
        }

        public event EventHandler CanExecuteChanged;

        protected virtual void OnCanExecuteChanged()
        {
            CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        }
    }
}
