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
    public class ResetCommand : ICommand
    {
        private readonly Models m_models;

        public ResetCommand(Models models)
        {
            m_models = models;
        }

        public bool CanExecute(object parameter)
        {
            return m_models.World != null && m_models.World.IsSane;
        }

        public void Execute(object parameter)
        {
            if (!Core.render_reset())
                throw new Exception(Core.core_get_dll_error());
            m_models.Renderer.updateIterationCount();
        }

        public event EventHandler CanExecuteChanged
        {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }
    }
}
