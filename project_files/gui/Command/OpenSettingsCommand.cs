using gui.View;
using gui.ViewModel;
using System;
using System.Windows.Input;
using gui.Model;
using gui.ViewModel.Settings;

namespace gui.Command
{
    class OpenSettingsCommand : ICommand
    {
        private readonly Model.Models m_models;
        private readonly SettingsViewModel m_viewModel;

        public OpenSettingsCommand(Models models)
        {
            m_models = models;
            m_viewModel = new SettingsViewModel(models);
        }

        public event EventHandler CanExecuteChanged
        {
            add {}
            remove {}
        }

        bool ICommand.CanExecute(object parameter)
        {
            return true; // TODO: change
        }

        void ICommand.Execute(object parameter)
        {
            // refresh view model
            m_viewModel.LoadFromSettings();

            var view = new SettingsView
            {
                DataContext = m_viewModel
            };

            if(m_models.App.ShowDialog(view) != true) return;

            // return value = true => save settings
            m_viewModel.StoreSettings();
        }
    }
}
