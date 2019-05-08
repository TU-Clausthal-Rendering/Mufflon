﻿using gui.Dll;
using gui.Model;
using System;
using System.Collections.Generic;
using System.ComponentModel;
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

        public RendererController(Models models)
        {
            m_models = models;
            m_models.App.GlHost.Loop += RenderLoop;
            m_models.App.GlHost.Destroy += OnDestroy;
        }
        
        /* Performs one render loop iteration
         * Blocks if no rendering, display update, or color picking is to be performed
         */
        private void RenderLoop(object sender, EventArgs args)
        {
            m_models.Statusbar.UpdateMemory();

            // Try to acquire the lock - if we're waiting, we're not rendering
            m_models.Renderer.RenderLock.WaitOne();
            // Check if we've been shut down
            if(m_shouldExit)
                return;

            UpdateRenderTargets();

            // This is necessary to avoid race conditions
            bool repaint = m_models.Renderer.IsRendering;

            if(repaint) {
                // Update camera movements and enter the render DLL
                if (m_models.Settings.AllowCameraMovement)
                    m_models.RendererCamera.MoveCamera();
                m_models.Renderer.Iterate();
            }

            // We release it to give the GUI a chance to block us (ie. rendering is supposed to pause/stop)
            m_models.Renderer.RenderLock.Release();
            if(repaint)
                m_models.Display.Repaint(m_renderTarget.TargetIndex, m_varianceTarget);
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
                    if (!Core.render_enable_render_target(newTarget.TargetIndex, newVarianceTarget ? 1u : 0u))
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.render_reset())
                        throw new Exception(Core.core_get_dll_error());
                } else if (!Core.render_is_render_target_enabled(newTarget.TargetIndex, newVarianceTarget)) {
                    // Disable previous render target
                    if (!Core.render_disable_render_target(m_renderTarget.TargetIndex, m_varianceTarget ? 1u : 0u))
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.render_enable_render_target(newTarget.TargetIndex, newVarianceTarget ? 1u : 0u))
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
            m_models.Renderer.RenderLock.Release();
        }
    }
}
