using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Dll;
using gui.Model;
using gui.Model.Camera;
using gui.Model.Light;

namespace gui.Command
{
    public class PlayPauseCommand : IGesturedCommand
    {
        private Models m_models;

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
            return m_models.Scene != null;
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
