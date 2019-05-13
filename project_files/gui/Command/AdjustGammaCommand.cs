using System;
using System.Windows.Input;
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
            m_models.Display.GammaFactor *= Factor;
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
            m_models.Display.GammaFactor /= Factor;
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
