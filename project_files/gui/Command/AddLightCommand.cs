using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using gui.Dll;
using gui.Model;
using gui.Model.Light;
using gui.View.Dialog;
using gui.ViewModel.Light;

namespace gui.Command
{
    public class AddLightCommand : ICommand
    {
        private readonly Models m_models;

        public AddLightCommand(Models models)
        {
            m_models = models;
        }

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public void Execute(object parameter)
        {
            var dc = new AddLightViewModel();
            var dialog = new AddPropertyDialog(dc);

            if (dialog.ShowDialog() != true) return;

            // TODO check for handle = null? error?
            var handle = Core.world_add_light(dc.NameValue, Core.FromModelLightType(dc.TypeValue));

            LightModel lm = LightModel.MakeFromHandle(handle, dc.TypeValue);

            m_models.World.Lights.Models.Add(lm);
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
