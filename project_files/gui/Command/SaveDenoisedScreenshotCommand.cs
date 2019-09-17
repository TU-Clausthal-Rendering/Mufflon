using System;
using System.IO;
using System.Windows;
using System.Windows.Input;
using gui.Dll;
using gui.Model;
using gui.Properties;

namespace gui.Command
{
    public class SaveDenoisedScreenshotCommand : ICommand
    {
        private readonly Models m_models;

        public SaveDenoisedScreenshotCommand(Models models)
        {
            m_models = models;
        }

        public bool CanExecute(object parameter)
        {
            // TODO: only screenshot when something was rendered?
            return m_models.World != null && m_models.World.Filename != null;
        }

        public void Execute(object parameter)
        {
            m_models.Renderer.TakeScreenshot(true);
        }

        public event EventHandler CanExecuteChanged
        {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }
    }
}
