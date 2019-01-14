using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Model;
using gui.Dll;

namespace gui.Command
{
    public class ResetCommand : IGesturedCommand
    {
        private Models m_models;
        private ICommand m_playPause;

        public ResetCommand(Models models, ICommand playPause) : base("ResetGesture")
        {
            m_models = models;
            m_playPause = playPause;
            CanExecuteChanged += OnCanExecuteChanged;
        }

        private void OnCanExecuteChanged(object sender, EventArgs args)
        {
            CommandManager.InvalidateRequerySuggested();
        }

        public override bool CanExecute(object parameter)
        {
            return m_models.Scene != null && m_models.Scene.IsSane;
        }

        public override void Execute(object parameter)
        {
            if (m_models.Renderer.IsRendering && m_playPause.CanExecute(null))
            {
                m_playPause.Execute(null);
                if (!Core.render_reset())
                    throw new Exception(Core.core_get_dll_error());
                m_playPause.Execute(null);
            } else
            {
                if (!Core.render_reset())
                    throw new Exception(Core.core_get_dll_error());
            }
            m_models.Renderer.Iteration = 0u;
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
