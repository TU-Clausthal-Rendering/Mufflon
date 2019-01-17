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
        }

        public override bool CanExecute(object parameter)
        {
            return m_models.World != null && m_models.World.IsSane;
        }

        public override void Execute(object parameter)
        {
            if (!Core.render_reset())
                throw new Exception(Core.core_get_dll_error());
            m_models.Renderer.updateIterationCount();
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
