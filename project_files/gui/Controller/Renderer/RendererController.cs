using gui.Command;
using gui.Dll;
using gui.Model;
using gui.Utility;
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
        private bool m_repaint = false;

        // When rendering animation sequences, this indicates that the next loop iteration should increment the animation frame
        private bool m_incrementFrame = false;
        // Tracks if we were the last animation frame in previous iteration (not discernible unfortunatel)
        private bool m_wasLastAnimFrame = false;

        private ManualResetEvent m_auxiliaryIteration = new ManualResetEvent(true);

        public RendererController(Models models)
        {
            m_models = models;
            m_models.App.GlHost.Loop += RenderLoop;
            m_models.App.GlHost.Destroy += OnDestroy;
            m_models.Renderer.RequestRedraw += OnRequestRedraw;
            m_models.Renderer.RequestWorldClear += OnRequestWorldClear;
            m_models.Renderer.PropertyChanged += LoadRendererParameters;
            m_models.Renderer.RequestParameterSave += SaveRendererParameter;
            m_models.Renderer.RequestScreenshot += OnRequestScreenshot;
            m_models.Display.RequestRepaint += OnRequestRepaint;
            m_models.Renderer.RequestScreenshot += OnRequestScreenshot;
            m_models.Display.RequestRepaint += OnRequestRepaint;
        }
        
        /* Performs one render loop iteration
         * Blocks if no rendering, display update, or color picking is to be performed
         */
        private void RenderLoop(object sender, EventArgs args)
        {
            m_models.Statusbar.UpdateMemory();

            // Try to acquire the lock - if we're waiting, we're not rendering
            if (m_shouldExit)
                return;
            m_models.Renderer.RenderLock.WaitOne();
            // Check if we've been shut down
            if(m_shouldExit)
                return;

            if (m_takeScreenshot)
            {
                string filename = ScreenShotCommand.ReplaceCommonFilenameTags(m_models, m_models.Settings.ScreenshotNamePattern);
                if (m_denoisedScreenshot)
                {
                    Core.render_save_denoised_radiance(Path.Combine(m_models.Settings.ScreenshotFolder, filename));
                }
                else
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
                if (!m_models.Renderer.IsRendering || m_iterationsPerformed == 0)
                    m_models.Renderer.RenderLock.Reset();
            } else if (m_clearWorld)
            {
                // Only clear the world, do nothing else
                Core.world_clear_all();
                m_clearWorld = false;
                m_models.Renderer.RenderLock.Reset();
                m_auxiliaryIteration.Set();
            } else
            {
                if(!m_repaint)
                {
                    UpdateRenderTargets();

                    if (m_models.Renderer.IsRendering)
                    {
                        if(m_incrementFrame)
                        {
                            m_incrementFrame = false;

                            if (m_models.Renderer.AnimationStart < m_models.World.AnimationFrameStart)
                            {
                                if (m_models.World.AnimationFrameCurrent >= m_models.World.AnimationFrameEnd)
                                    m_models.World.AnimationFrameCurrent = m_models.World.AnimationFrameStart;
                                else
                                    ++m_models.World.AnimationFrameCurrent;
                            } else
                            {
                                if(m_models.World.AnimationFrameCurrent >= m_models.Renderer.AnimationEnd)
                                {
                                    m_iterationsPerformed = m_models.Renderer.RemainingIterations;
                                    m_wasLastAnimFrame = true;
                                }
                                ++m_models.World.AnimationFrameCurrent;
                            }
                            m_iterationsPerformed = 0;
                        }

                        if (m_iterationsPerformed < m_models.Renderer.RemainingIterations || m_models.Renderer.RemainingIterations < 0)
                        {
                            // Update camera movements and enter the render DLL
                            if (m_models.Settings.AllowCameraMovement)
                                m_models.RendererCamera.MoveCamera();
                            m_models.Renderer.Iterate();
                            ++m_iterationsPerformed;
                        }
                        else
                        {
                            if(m_models.Renderer.AnimationFrameComplete != null && !m_wasLastAnimFrame)
                                m_models.Renderer.AnimationFrameComplete();

                            // Check if we're animating
                            if (m_models.Renderer.RenderAnimation)
                            {
                                // Check if we need to reset the animation frame
                                if (m_models.Renderer.AnimationStart < m_models.World.AnimationFrameStart)
                                {
                                    m_incrementFrame = true;
                                } else
                                {
                                    // Check if we should stop
                                    if(m_models.World.AnimationFrameCurrent > m_models.Renderer.AnimationEnd || m_wasLastAnimFrame)
                                    {
                                        m_wasLastAnimFrame = false;
                                        System.Windows.Application.Current.Dispatcher.Invoke(new Action(() => m_models.Renderer.IsRendering = false));
                                        m_models.Renderer.RenderLock.Reset();
                                    } else
                                    {
                                        m_incrementFrame = true;
                                    }
                                }
                            } else
                            {
                                // No more iterations left -> we're done
                                System.Windows.Application.Current.Dispatcher.Invoke(new Action(() => m_models.Renderer.IsRendering = false));
                                m_models.Renderer.RenderLock.Reset();
                            }
                        }
                    }
                    else
                    {
                        m_iterationsPerformed = 0;
                        m_models.Renderer.RenderLock.Reset();
                    }

                    // We release it to give the GUI a chance to block us (ie. rendering is supposed to pause/stop)
                    if (m_shouldExit)
                        return;
                }

                // Renew the display texture (we don't really care)
                IntPtr targetTexture = IntPtr.Zero;
                if (!Core.core_get_target_image(m_renderTarget.Name, m_varianceTarget, out targetTexture))
                    throw new Exception(Core.core_get_dll_error());
                m_models.Display.Repaint(!m_repaint);

                if (m_repaint)
                {
                    m_repaint = false;
                    if (!m_models.Renderer.IsRendering || m_iterationsPerformed == 0)
                        m_models.Renderer.RenderLock.Reset();
                    m_auxiliaryIteration.Set();
                }
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

        private void OnRequestRepaint()
        {
            m_auxiliaryIteration.Reset();
            m_repaint = true;
            m_models.Renderer.RenderLock.Set();
            m_auxiliaryIteration.WaitOne();
        }

        // Saves the current renderer's parameters in textform in the application settings
        private void SaveRendererParameter(RendererParameter param)
        {
            ObservableDictionary<string, SerializedRendererParameter> rendererParams;
            if(!m_models.Settings.RendererParameters.TryGetValue(m_models.Renderer.Name, out rendererParams))
            {
                rendererParams = new ObservableDictionary<string, SerializedRendererParameter>();
                m_models.Settings.RendererParameters.Add(m_models.Renderer.Name, rendererParams);
            }
            SerializedRendererParameter storedParam = new SerializedRendererParameter();
            storedParam.Type = param.Type;
            storedParam.Value = param.Value.ToString();
            rendererParams[param.Name] = storedParam;
        }

        // Restores the current renderer's parameters from the application settings
        private void LoadRendererParameters(object sender, PropertyChangedEventArgs args)
        {
            if(args.PropertyName == nameof(Model.RendererModel.Name))
            {
                ObservableDictionary<string, SerializedRendererParameter> rendererParams;
                if (m_models.Settings.RendererParameters.TryGetValue(m_models.Renderer.Name, out rendererParams))
                {
                    foreach (var param in m_models.Renderer.Parameters)
                    {
                        SerializedRendererParameter storedParam;
                        if (rendererParams.TryGetValue(param.Name, out storedParam) && storedParam.Type == param.Type)
                        {
                            switch(param.Type)
                            {
                                case Core.ParameterType.Bool: {
                                    bool val;
                                    if (Boolean.TryParse(storedParam.Value, out val))
                                        param.Value = val;
                                }   break;
                                case Core.ParameterType.Int: {
                                    Int32 val;
                                    if (Int32.TryParse(storedParam.Value, out val))
                                        param.Value = val;
                                }   break;
                                case Core.ParameterType.Float: {
                                    double val;
                                    if (Double.TryParse(storedParam.Value, out val))
                                        param.Value = (float)val;
                                }   break;
                                case Core.ParameterType.Enum:
                                    param.Value = storedParam.Value;
                                    break;
                            }
                        }
                    }
                }
            }
        }
    }
}
