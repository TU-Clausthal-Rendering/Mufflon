using System;
using System.IO;
using System.Windows;
using System.Windows.Input;
using gui.Dll;
using gui.Model;
using gui.Properties;

namespace gui.Command
{
    public class ScreenShotCommand : ICommand
    {
        private readonly Models m_models;

        public ScreenShotCommand(Models models)
        {
            m_models = models;
        }

        public static string ReplaceCommonFilenameTags(Models models, string filename)
        {
            return filename.Replace("#scene", Path.GetFileNameWithoutExtension(models.World.Filename));
        }

        public bool CanExecute(object parameter)
        {
            // TODO: only screenshot when something was rendered?
            return m_models.World != null && m_models.World.Filename != null;
        }

        public void Execute(object parameter)
        {
            // First parse the current screenshot string and emplace the information
            string filename = ReplaceCommonFilenameTags(m_models, m_models.Settings.ScreenshotNamePattern);

            foreach (RenderTarget target in m_models.RenderTargetSelection.Targets)
            {
                if (target.Enabled)
                    Core.render_save_screenshot(Path.Combine(m_models.Settings.ScreenshotFolder, filename), target.Name, false);
                if (target.VarianceEnabled)
                    Core.render_save_screenshot(Path.Combine(m_models.Settings.ScreenshotFolder, filename), target.Name, true);
            }
        }

        public event EventHandler CanExecuteChanged
        {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }
    }
}
