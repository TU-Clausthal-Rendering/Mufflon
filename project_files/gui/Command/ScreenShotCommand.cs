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

        public bool CanExecute(object parameter)
        {
            // TODO: only screenshot when something was rendered?
            return m_models.World != null && m_models.World.Filename != null;
        }

        public void Execute(object parameter)
        {
            // First parse the current screenshot string and emplace the information
            string filename = m_models.Settings.ScreenshotNamePattern;
            filename = filename.Replace("#scene", Path.GetFileNameWithoutExtension(m_models.World.Filename));
            filename = filename.Replace("#scenario", m_models.World.CurrentScenario.Name);
            filename = filename.Replace("#renderer", Core.render_get_renderer_name(m_models.Renderer.RendererIndex));
            filename = filename.Replace("#iteration", m_models.Renderer.Iteration.ToString());
            filename = filename.Replace("#target", RenderTargetSelectionModel.getRenderTargetName(m_models.RenderTargetSelection.VisibleTarget,
                m_models.RenderTargetSelection.IsVarianceVisible));
            Core.render_save_screenshot(Path.Combine(m_models.Settings.ScreenshotFolder, filename));
        }

        public event EventHandler CanExecuteChanged
        {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }
    }
}
