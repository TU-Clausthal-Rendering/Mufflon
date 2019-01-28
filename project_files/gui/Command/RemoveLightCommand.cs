using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Dll;
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
            var lightName = m_model.Name;
            if(!Core.world_remove_light(m_model.Handle))
                throw new Exception(Core.core_get_dll_error());

            m_models.World.Lights.Models.Remove(m_model);

            // was the light in the current scenario?
            

            // TODO why do I need to do this? (why isn't this done by world_remove_light?)
            var currScenario = Core.world_get_current_scenario();
            if (!Core.scenario_remove_light(currScenario, m_model.Handle))
                throw new Exception(Core.core_get_dll_error());
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
