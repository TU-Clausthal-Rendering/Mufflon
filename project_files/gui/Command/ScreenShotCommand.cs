using System;
using System.IO;
using System.Windows;
using System.Windows.Input;
using gui.Dll;
using gui.Model;
using gui.Properties;

namespace gui.Command
{
    public class ScreenShotCommand : IGesturedCommand
    {
        private Models m_models;

        public ScreenShotCommand(Models models) : base("ScreenshotGesture")
        {
            m_models = models;
        }

        public override bool CanExecute(object parameter)
        {
            // TODO: only screenshot when something was rendered?
            return true;
        }

        public override void Execute(object parameter)
        {
            // First parse the current screenshot string and emplace the information
            string filename = Settings.Default.ScreenshotNamePattern;
            filename = filename.Replace("#scene", Path.GetFileNameWithoutExtension(m_models.Scene.Filename));
            filename = filename.Replace("#scenario", m_models.Scene.CurrentScenario);
            filename = filename.Replace("#renderer", RendererModel.getRendererName(m_models.Renderer.Type));
            filename = filename.Replace("#iteration", m_models.Renderer.Iteration.ToString());
            // Gotta pause the renderer
            bool wasRunning = m_models.Renderer.IsRendering;
            m_models.Renderer.IsRendering = false;
            Core.render_save_screenshot(Path.Combine(Settings.Default.ScreenshotFolder, filename));
            m_models.Renderer.IsRendering = wasRunning;
        }

        public override event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
