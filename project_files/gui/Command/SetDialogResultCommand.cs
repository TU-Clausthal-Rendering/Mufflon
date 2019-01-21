using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Model;

namespace gui.Command
{
    /// <summary>
    /// sets the dialog result of the topmost window (which closes the dialog)
    /// </summary>
    public class SetDialogResultCommand : ICommand
    {
        private readonly Models m_models;
        private readonly bool m_result;

        public SetDialogResultCommand(Models models, bool result)
        {
            m_models = models;
            m_result = result;
        }

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public void Execute(object parameter)
        {
            m_models.App.TopmostWindow.DialogResult = m_result;
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
