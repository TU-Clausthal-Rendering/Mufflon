using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Model;
using gui.Model.Light;

namespace gui.Command
{
    public class RemoveLightCommand : ICommand
    {
        private readonly Models m_models;
        private readonly LightModel m_model;

        public RemoveLightCommand(Models models, LightModel model)
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
            m_models.World.Lights.Models.Remove(m_model);
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
