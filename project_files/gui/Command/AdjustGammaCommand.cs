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
        public float Factor = 2f;

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public void Execute(object parameter)
        {
            OpenGlDisplay.opengldisplay_set_gamma(OpenGlDisplay.opengldisplay_get_gamma() * Factor);
        }

        public event EventHandler CanExecuteChanged
        {
            add{}
            remove{}
        }
    }

    public class AdjustGammaDownCommand : ICommand
    {
        public float Factor = 2f;

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public void Execute(object parameter)
        {
            OpenGlDisplay.opengldisplay_set_gamma(OpenGlDisplay.opengldisplay_get_gamma() / Factor);
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
