using gui.View;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace gui.Command
{
    class OpenSettingsCommand : ICommand
    {
        private AppSettings m_settings;
        event EventHandler ICommand.CanExecuteChanged
        {
            add
            {
            }

            remove
            {
            }
        }

        bool ICommand.CanExecute(object parameter)
        {
            return true; // TODO: change
        }

        void ICommand.Execute(object parameter)
        {
            m_settings = new AppSettings();
            m_settings.Show();
        }
    }
}
