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
            filename = filename.Replace("#scene", Path.GetFileNameWithoutExtension(models.World.Filename));
            filename = filename.Replace("#scenario", models.World.CurrentScenario.Name);
            filename = filename.Replace("#renderer", models.Renderer.Name);
            filename = filename.Replace("#shortrenderer", models.Renderer.ShortName);
            filename = filename.Replace("#iteration", models.Renderer.Iteration.ToString());

            // Enumerate devices
            string devs = "";
            foreach (Core.RenderDevice dev in Enum.GetValues(typeof(Core.RenderDevice)))
            {
                if (models.Renderer.UsesDevice(dev))
                    devs += Enum.GetName(typeof(Core.RenderDevice), dev) + ",";
            }
            if (devs.Length > 0)
                devs = devs.Substring(0, devs.Length - 1);
            return filename.Replace("#devices", devs);
        }

        public static string ReplaceTargetFilenameTags(Models models, string name, bool variance, string filename)
        {
            if (variance)
                return filename.Replace("#target", name + "(Variance)");
            else
                return filename.Replace("#target", name);
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
                {
                    string file = ReplaceTargetFilenameTags(m_models, target.Name, false, filename);
                    Core.render_save_screenshot(Path.Combine(m_models.Settings.ScreenshotFolder, file), target.Name, false);
                }
                if (target.VarianceEnabled)
                {
                    string file = ReplaceTargetFilenameTags(m_models, target.Name, true, filename);
                    Core.render_save_screenshot(Path.Combine(m_models.Settings.ScreenshotFolder, file), target.Name, true);
                }
            }
        }

        public event EventHandler CanExecuteChanged
        {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }
    }
}
