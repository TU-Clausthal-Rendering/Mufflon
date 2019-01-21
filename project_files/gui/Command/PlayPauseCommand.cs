using System;
using System.ComponentModel;
using System.Threading;
using System.Windows.Input;
using gui.Model;

namespace gui.Command
{
    public class PlayPauseCommand : ICommand
    {
        private Models m_models;

        public PlayPauseCommand(Models models)
        {
            m_models = models;
        }

        public bool CanExecute(object parameter)
        {
            return m_models.World != null && m_models.World.IsSane;
        }

        public void Execute(object parameter)
        {
            m_models.Renderer.IsRendering = !m_models.Renderer.IsRendering;
        }

        public event EventHandler CanExecuteChanged
        {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }
    }
}
