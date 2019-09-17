using gui.Command;
using gui.Dll;
using gui.Model;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace gui.Controller.Renderer
{
    public class RendererController
    {
        private Models m_models;
        // Store the selected render targets to not get into trouble with multithreading
        private RenderTarget m_renderTarget;
        private bool m_varianceTarget;
        // Store the render size to avoid issues with multithreading; this gets updated once per render iteration
        private int m_renderWidth, m_renderHeight;
        // Track if the thread should terminate
        private bool m_shouldExit = false;

        // Keep track of the number of consecutive iterations performed
        private int m_iterationsPerformed = 0;
        // Check if we got called to clear the world
        private bool m_clearWorld = false;
        private bool m_takeScreenshot = false;
        private bool m_denoisedScreenshot = false;
        private ManualResetEvent m_auxiliaryIteration = new ManualResetEvent(true);

        public RendererController(Models models)
        {
            m_models = models;
            m_models.App.GlHost.Loop += RenderLoop;
            m_models.App.GlHost.Destroy += OnDestroy;
            m_models.Renderer.RequestRedraw += OnRequestRedraw;
            m_models.Renderer.RequestWorldClear += OnRequestWorldClear;
            m_models.Renderer.PropertyChanged += LoadRendererParameters;
            m_models.Renderer.RequestParameterSave += SaveRendererParameters;
            m_models.Renderer.RequestScreenshot += OnRequestScreenshot;
        }
        
        /* Performs one render loop iteration
         * Blocks if no rendering, display update, or color picking is to be performed
         */
        private void RenderLoop(object sender, EventArgs args)
        {
            m_models.Statusbar.UpdateMemory();

            // Try to acquire the lock - if we're waiting, we're not rendering
            if(m_shouldExit)
                return;
            m_models.Renderer.RenderLock.WaitOne();
            // Check if we've been shut down
            if(m_shouldExit)
                return;

            if(m_takeScreenshot)
            {
                string filename = ScreenShotCommand.ReplaceCommonFilenameTags(m_models, m_models.Settings.ScreenshotNamePattern);
                if(m_denoisedScreenshot)
                {
                    Core.render_save_denoised_radiance(Path.Combine(m_models.Settings.ScreenshotFolder, filename));
                } else
                {
                    foreach (RenderTarget target in m_models.RenderTargetSelection.Targets)
                    {
                        if (target.Enabled)
                            Core.render_save_screenshot(Path.Combine(m_models.Settings.ScreenshotFolder, filename), target.Name, false);
                        if (target.VarianceEnabled)
                            Core.render_save_screenshot(Path.Combine(m_models.Settings.ScreenshotFolder, filename), target.Name, true);
                    }
                }

                m_takeScreenshot = false;
                if(!m_models.Renderer.IsRendering || m_iterationsPerformed == 0)
                    m_models.Renderer.RenderLock.Reset();
            } else if(m_clearWorld)
            {
                // Only clear the world, do nothing else
                Core.world_clear_all();
                m_clearWorld = false;
                m_models.Renderer.RenderLock.Reset();
                m_auxiliaryIteration.Set();
            } else
            {
                UpdateRenderTargets();

                if(m_models.Renderer.IsRendering) {
                    if(m_iterationsPerformed < m_models.Renderer.RemainingIterations || m_models.Renderer.RemainingIterations < 0)
                    {
                        // Update camera movements and enter the render DLL
                        if (m_models.Settings.AllowCameraMovement)
                            m_models.RendererCamera.MoveCamera();
                        m_models.Renderer.Iterate();
                        ++m_iterationsPerformed;
                    } else
                    {
                        // No more iterations left -> we're done
                        m_iterationsPerformed = 0;
                        System.Windows.Application.Current.Dispatcher.Invoke(new Action(() => m_models.Renderer.IsRendering = false));
                        m_models.Renderer.RenderLock.Reset();
                    }
                } else {
                    m_iterationsPerformed = 0;
                    m_models.Renderer.RenderLock.Reset();
                }

                // We release it to give the GUI a chance to block us (ie. rendering is supposed to pause/stop)
                if(m_shouldExit)
                    return;
                m_models.Display.Repaint(m_renderTarget.Name, m_varianceTarget);
            }
        }

        // Handles changes in viewport size and render targets
        private void UpdateRenderTargets()
        {
            // Check if we need to resize the screen texture
            int newWidth = m_models.Display.RenderSize.X;
            int newHeight = m_models.Display.RenderSize.Y;
            RenderTarget newTarget = m_models.RenderTargetSelection.VisibleTarget;
            bool newVarianceTarget = m_models.RenderTargetSelection.IsVarianceVisible;

            // TODO: disable the old target?
            if(newTarget != m_renderTarget || newVarianceTarget != m_varianceTarget) {
                if(m_renderTarget == null) {
                    if (!Core.render_enable_render_target(newTarget.Name, newVarianceTarget))
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.render_reset())
                        throw new Exception(Core.core_get_dll_error());
                } else if (!Core.render_is_render_target_enabled(newTarget.Name, newVarianceTarget)) {
                    // Disable previous render target
                    if (!Core.render_disable_render_target(m_renderTarget.Name, m_varianceTarget))
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.render_enable_render_target(newTarget.Name, newVarianceTarget))
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.render_reset())
                        throw new Exception(Core.core_get_dll_error());
                }
            }


            if (m_renderWidth != newWidth || m_renderHeight != newHeight || newTarget != m_renderTarget
                || newVarianceTarget != m_varianceTarget) {
                m_renderWidth = newWidth;
                m_renderHeight = newHeight;
                m_renderTarget = newTarget;
                m_varianceTarget = newVarianceTarget;
            }
        }

        private void OnDestroy(object sender, EventArgs args) {
            // Tell our loop
            m_shouldExit = true;
            // This needs to happen in the main UI thread
            //if(!m_models.Renderer.IsRendering)
            m_models.Renderer.RenderLock.Set();
        }

        private void OnRequestRedraw(object sender, EventArgs args)
        {
            m_models.Renderer.RenderLock.Set();
        }

        private void OnRequestWorldClear(object sender, EventArgs args)
        {
            m_auxiliaryIteration.Reset();
            m_clearWorld = true;
            m_models.Renderer.RenderLock.Set();
            m_auxiliaryIteration.WaitOne();
        }

        private void OnRequestScreenshot(bool denoised)
        {
            m_denoisedScreenshot = denoised;
            m_takeScreenshot = true;
            m_models.Renderer.RenderLock.Set();
        }


        private int FindRendererDictionaryIndex(string name)
        {
            if (m_models.Settings.RendererParameters.Count % 2 != 0)
            {
                Logger.log("Invalid saved renderer properties; resetting all renderer properties to defaults", Core.Severity.Error);
                m_models.Settings.RendererParameters.Clear();
            }
            for (int i = 0; i < m_models.Settings.RendererParameters.Count; i += 2)
            {
                if (m_models.Settings.RendererParameters[i].Equals(name))
                    return i + 1;
            }
            return -1;
        }
        private string getSettingsKey()
        {
            string key = m_models.Renderer.Name;
            foreach (Core.RenderDevice dev in Enum.GetValues(typeof(Core.RenderDevice)))
            {
                if (m_models.Renderer.UsesDevice(dev))
                    key += "-" + Enum.GetName(typeof(Core.RenderDevice), dev);
            }
            return key;
        }

        // Saves the current renderer's parameters in textform in the application settings
        private void SaveRendererParameters(object sender, EventArgs ignored)
        {
            if (m_models.Renderer.Parameters == null)
                return;

            string val = "";

            foreach (var param in m_models.Renderer.Parameters)
                val += param.Name + ";" + Enum.GetName(typeof(Core.ParameterType), param.Type) + ";" + param.Value.ToString() + "\n";

            string key = getSettingsKey();
            int idx = FindRendererDictionaryIndex(key);
            if (idx < 0)
            {
                m_models.Settings.RendererParameters.Add(key);
                m_models.Settings.RendererParameters.Add(val);
            }
            else
            {
                m_models.Settings.RendererParameters[idx] = val;
            }
        }

        // Restores the current renderer's parameters from the application settings
        private void LoadRendererParameters(object sender, PropertyChangedEventArgs args)
        {
            if(args.PropertyName == nameof(Model.RendererModel.Name))
            {
                string key = getSettingsKey();
                int idx = FindRendererDictionaryIndex(key);
                if (idx < 0)
                {
                    m_models.Settings.RendererParameters.Add(key);
                    m_models.Settings.RendererParameters.Add("");
                }
                else
                {
                    string parameters = m_models.Settings.RendererParameters[idx];

                    m_models.Renderer.LoadRendererParametersFromString(parameters);
                }
            }
        }
    }
}
