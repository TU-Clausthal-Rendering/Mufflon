using System;
using System.Windows.Input;
using gui.Model;
using gui.View.Dialog;
using gui.ViewModel;

namespace gui.Command
{
    public class SelectRendererCommand : ICommand
    {
        private readonly Models m_models;

        public SelectRendererCommand(Models models)
        {
            m_models = models;
        }

        public bool CanExecute(object parameter)
        {
            return !m_models.Renderer.IsRendering;
        }

        public virtual void Execute(object parameter)
        {
            var data = new SelectRendererViewModel();
            var dialog = new SelectRendererDialog(data);
            if (dialog.ShowDialog() != true) return;
            m_models.Renderer.Type = data.TypeValue;
            // TODO: display the current renderer somewhere
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
