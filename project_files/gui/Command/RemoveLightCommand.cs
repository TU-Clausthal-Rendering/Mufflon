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
        private readonly ICommand m_reset;

        public RemoveLightCommand(Models models, LightModel model)
        {
            m_models = models;
            m_model = model;
            // TODO replace this with something better...
            m_reset = new ResetCommand(models, new PlayPauseCommand(models));
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
            if (!m_model.IsSelected) return;

            // TODO why do I need to do this? (why isn't this done by world_remove_light?)
            var currScenario = Core.world_get_current_scenario();
            if (!Core.scenario_remove_light(currScenario, lightName))
                throw new Exception(Core.core_get_dll_error());

            // scene needs to be rebuilded
            bool wasRendering = m_models.Renderer.IsRendering;
            m_models.Renderer.IsRendering = false;

            if(m_reset.CanExecute(null))
                m_reset.Execute(null);

            if (Core.world_reload_current_scenario() == IntPtr.Zero)
                throw new Exception(Core.core_get_dll_error());
            m_models.Renderer.IsRendering = wasRendering;
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
