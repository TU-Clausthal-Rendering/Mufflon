using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
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

            LightModel lm = null;
            switch (dc.TypeValue)
            {
                case LightModel.LightType.Point:
                    lm = new PointLightModel();
                    break;
                case LightModel.LightType.Directional:
                    lm = new DirectionalLightModel();
                    break;
                case LightModel.LightType.Spot:
                    lm = new SpotLightModel();
                    break;
                case LightModel.LightType.Envmap:
                    lm = new EnvmapLightModel();
                    break;
                case LightModel.LightType.Goniometric:
                    lm = new GoniometricLightModel();
                    break;
            }
            Debug.Assert(lm != null);

            lm.Name = dc.NameValue;

            m_models.Lights.Models.Add(lm);
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
