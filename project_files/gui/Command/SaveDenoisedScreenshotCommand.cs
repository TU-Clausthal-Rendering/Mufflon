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
            string filename = ScreenShotCommand.ReplaceCommonFilenameTags(m_models, m_models.Settings.ScreenshotNamePattern);
            Dll.Core.render_save_denoised_radiance(Path.Combine(m_models.Settings.ScreenshotFolder, filename));
        }

        public event EventHandler CanExecuteChanged
        {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }
    }
}
