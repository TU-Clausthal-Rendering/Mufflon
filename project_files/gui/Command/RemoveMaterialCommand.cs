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
        private readonly Models m_models;
        private readonly MaterialModel m_model;

        public RemoveMaterialCommand(Models models, MaterialModel model)
        {
            m_models = models;
            m_model = model;
        }

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public void Execute(object parameter)
        {
            m_models.Materials.Models.Remove(m_model);
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
