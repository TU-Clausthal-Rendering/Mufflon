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
            string filename = Settings.Default.ScreenshotNamePattern;
            filename = filename.Replace("#scene", Path.GetFileNameWithoutExtension(m_models.World.Filename));
            filename = filename.Replace("#scenario", m_models.World.CurrentScenario.Name);
            filename = filename.Replace("#renderer", RendererModel.getRendererName(m_models.Renderer.Type));
            filename = filename.Replace("#iteration", m_models.Renderer.Iteration.ToString());
            filename = filename.Replace("#target", RendererModel.getRenderTargetName(m_models.Renderer.RenderTarget,
                m_models.Renderer.RenderTargetVariance));
            // Gotta pause the renderer
            bool wasRunning = m_models.Renderer.IsRendering;
            m_models.Renderer.IsRendering = false;
            Core.render_save_screenshot(Path.Combine(Settings.Default.ScreenshotFolder, filename));
            m_models.Renderer.IsRendering = wasRunning;
        }

        public event EventHandler CanExecuteChanged
        {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }
    }
}
