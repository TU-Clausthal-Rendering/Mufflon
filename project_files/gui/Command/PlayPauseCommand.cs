using System;
using System.ComponentModel;
using System.Threading;
using System.Windows.Input;
using gui.Model;

namespace gui.Command
{
    public class PlayPauseCommand : IGesturedCommand
    {
        private Models m_models;

        public PlayPauseCommand(Models models) : base("PlayPauseGesture")
        {
            m_models = models;
        }

        public override bool CanExecute(object parameter)
        {
            return m_models.World != null && m_models.World.IsSane;
        }

        public override void Execute(object parameter)
        {
            m_models.Renderer.IsRendering = !m_models.Renderer.IsRendering;
        }

        public override event EventHandler CanExecuteChanged
        {
            add
            {
                CommandManager.RequerySuggested += value;
            }
            remove
            {
                CommandManager.RequerySuggested -= value;
            }
        }
    }
}
