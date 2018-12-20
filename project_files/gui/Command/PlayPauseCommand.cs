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
    public class PlayPauseCommand : ICommand
    {
        private Models m_models;
        private volatile bool m_finishedIteration = true;

        public PlayPauseCommand(Models models)
        {
            m_models = models;
            CanExecuteChanged += OnCanExecuteChanged;
            m_models.Renderer.PropertyChanged += OnIterationComplete;
        }

        private void OnCanExecuteChanged(object sender, EventArgs args)
        {
            CommandManager.InvalidateRequerySuggested();
        }

        public bool CanExecute(object parameter)
        {
            return m_models.Scene.IsLoaded;
        }

        public void Execute(object parameter)
        {
            m_models.Renderer.IsRendering = !m_models.Renderer.IsRendering;
            if(!m_models.Renderer.IsRendering)
            {
                m_finishedIteration = false;
                // Wait until we get signaled that the iteration is over (with a timeout for bad cases - 10 seconds should suffice)
                var task = Task.Run(() => { while (!m_finishedIteration) ; });
                task.Wait(TimeSpan.FromSeconds(10));
            }
        }

        private void OnIterationComplete(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(RendererModel.Iteration):
                    m_finishedIteration = true;
                    break;
            }
        }

        public event EventHandler CanExecuteChanged
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
