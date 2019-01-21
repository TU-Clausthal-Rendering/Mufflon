using gui.View;
using gui.ViewModel;
using System;
using System.Windows.Input;
using gui.Model;

namespace gui.Command
{
    class OpenSettingsCommand : ICommand
    {
        private readonly Model.Models m_models;
        private readonly ViewModels m_viewModels;
        private AppSettings m_settings;

        public OpenSettingsCommand(ViewModels viewModels, Models models)
        {
            m_viewModels = viewModels;
            m_models = models;
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
            m_settings = new AppSettings(m_viewModels, m_models);
            m_settings.Show();
        }
    }
}
