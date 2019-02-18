using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Dll;
using gui.Model;

namespace gui.Command
{
    public class AdjustGammaUpCommand : ICommand
    {
        private Models m_models;
        public float Factor = 2f;

        public AdjustGammaUpCommand(Models models)
        {
            m_models = models;
        }

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public void Execute(object parameter)
        {
            OpenGlDisplay.opengldisplay_set_factor(OpenGlDisplay.opengldisplay_get_factor() * Factor);
            m_models.Renderer.UpdateDisplayTexture();
        }

        public event EventHandler CanExecuteChanged
        {
            add{}
            remove{}
        }
    }

    public class AdjustGammaDownCommand : ICommand
    {
        private Models m_models;
        public float Factor = 2f;

        public AdjustGammaDownCommand(Models models)
        {
            m_models = models;
        }

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public void Execute(object parameter)
        {
            OpenGlDisplay.opengldisplay_set_factor(OpenGlDisplay.opengldisplay_get_factor() / Factor);
            m_models.Renderer.UpdateDisplayTexture();
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
