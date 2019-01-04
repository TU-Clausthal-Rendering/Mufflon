using gui.View;
using gui.ViewModel;
using System;
using System.Windows.Input;

namespace gui.Command
{
    class OpenSettingsCommand : ICommand
    {
        private ViewModels m_viewModels;
        private AppSettings m_settings;

        public OpenSettingsCommand(ViewModels viewModels)
        {
            m_viewModels = viewModels;
        }

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
            m_settings = new AppSettings(m_viewModels);
            m_settings.Show();
        }
    }
}
