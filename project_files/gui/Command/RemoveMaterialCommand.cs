using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Model;
using gui.Model.Material;

namespace gui.Command
{
    public class RemoveMaterialCommand : ICommand
    {
        private readonly MaterialModel m_model;

        public RemoveMaterialCommand(MaterialModel model)
        {
            m_model = model;
        }

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public void Execute(object parameter)
        {
            m_model.Remove();
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
