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
        private ManualResetEvent m_iterationComplete = new ManualResetEvent(false);

        public PlayPauseCommand(Models models) : base("PlayPauseGesture")
        {
            m_models = models;
            CanExecuteChanged += OnCanExecuteChanged;
        }

        private void OnCanExecuteChanged(object sender, EventArgs args)
        {
            CommandManager.InvalidateRequerySuggested();
        }

        public override bool CanExecute(object parameter)
        {
            return m_models.Scene.IsLoaded && m_models.Scene.IsSane;
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
