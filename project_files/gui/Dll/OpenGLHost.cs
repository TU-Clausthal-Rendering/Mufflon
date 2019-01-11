﻿using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using gui.Model;
using gui.Properties;
using gui.Utility;
using gui.ViewModel;

namespace gui.Dll
{
    public class OpenGLHost : HwndHost
    {
        // error callbacks (in case the openGL thread crashes)
        public delegate void ErrorEvent(string message);
        public event ErrorEvent Error;

        // host of the HwndHost
        private readonly MainWindow m_window;
        private readonly Border m_parent;
        // information about the viewport
        private readonly ViewportModel m_viewport;
        private readonly RendererModel m_rendererModel;

        // context creation
        private IntPtr m_hWnd = IntPtr.Zero;
        private IntPtr m_deviceContext = IntPtr.Zero;

        // render thread (asynchronous openGL drawing)
        private Thread m_renderThread;
        private bool m_isRunning = true;

        // Blocker for when nothing is rendering to avoid busy-loop
        private readonly ConcurrentQueue<string> m_commandQueue = new ConcurrentQueue<string>();

        // helper to detect resize in render thread
        private int m_renderWidth = 0;
        private int m_renderHeight = 0;
        private int m_renderOffsetX = 0;
        private int m_renderOffsetY = 0;

        // TODO: this is only for testing purposes
        public static bool toggleRenderer = false;

        // this is required to prevent the callback from getting garbage collected
        private Core.LogCallback m_logCallbackPointer = null;

        public OpenGLHost(MainWindow window, ViewportModel viewport, RendererModel rendererModel)
        {
            m_window = window;
            m_parent = window.BorderHost;
            m_viewport = viewport;
            m_rendererModel = rendererModel;
        }

        /// <summary>
        /// queues commands for the render thread.
        /// The commands will be redirected to the dll
        /// </summary>
        /// <param name="command"></param>
        public void QueueCommand(string command)
        {
            m_commandQueue.Enqueue(command);
        }

        /// <summary>
        /// asynch thread: render method
        /// </summary>
        /// <param name="data"></param>
        private void Render(object data)
        {
            // TODO: make these settings
            float keySpeed = 0.125f;
            float mouseSpeed = 0.0025f;

            try
            {
                InitializeOpenGl();
                Logger.LogLevel = (Core.Severity)Settings.Default.LogLevel;
                Core.profiling_enable();
                Loader.loader_profiling_enable();
                if (!Core.profiling_set_level((Core.ProfilingLevel)Settings.Default.CoreProfileLevel))
                    throw new Exception(Core.core_get_dll_error());
                if (!Loader.loader_profiling_set_level((Core.ProfilingLevel)Settings.Default.LoaderProfileLevel))
                    throw new Exception(Loader.loader_get_dll_error());

                while (m_isRunning)
                {

                    //HandleCommands();
                    HandleResize();
                    // Try to acquire the lock - if we're waiting, we're not rendering
                    m_rendererModel.RenderLock.WaitOne();

                    bool needsReload = false;

                    // Check for keyboard input
                    float x = 0f;
                    float y = 0f;
                    float z = 0f;
                    // TODO: why does it need to be mirrored?
                    if(m_window.wasPressedAndClear(Key.W))
                        z += keySpeed;
                    if (m_window.wasPressedAndClear(Key.S))
                        z -= keySpeed;
                    if (m_window.wasPressedAndClear(Key.D))
                        x -= keySpeed;
                    if (m_window.wasPressedAndClear(Key.A))
                        x += keySpeed;
                    if (m_window.wasPressedAndClear(Key.Space))
                        y += keySpeed;
                    if (m_window.wasPressedAndClear(Key.LeftCtrl))
                        y -= keySpeed;

                    if(x != 0f || y != 0f || z != 0f) {
                        // TODO: mark camera dirty
                        if (!Core.scene_move_active_camera(x, y, z))
                            throw new Exception(Core.core_get_dll_error());
                        needsReload = true;
                    }

                    // Check for mouse dragging
                    Vector drag = m_window.getMouseDiffAndReset();
                    if(drag.X != 0 || drag.Y != 0)
                    {
                        if(!Core.scene_rotate_active_camera(mouseSpeed * (float)drag.Y, -mouseSpeed * (float)drag.X, 0))
                            throw new Exception(Core.core_get_dll_error());
                        needsReload = true;
                    }

                    // Reload the scene if necessary
                    if(needsReload)
                    {
                        if (Core.world_reload_current_scenario() == IntPtr.Zero)
                            throw new Exception(Core.core_get_dll_error());
                        if (!Core.render_reset())
                            throw new Exception(Core.core_get_dll_error());
                        m_rendererModel.Iteration = 0u;
                    }

                    if (!Core.render_iterate())
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.display_screenshot())
                        throw new Exception(Core.core_get_dll_error());
                    if (!Gdi32.SwapBuffers(m_deviceContext))
                        throw new Win32Exception(Marshal.GetLastWin32Error());
                    ++m_rendererModel.Iteration;

                    // We release it to give the GUI a chance to block us (ie. rendering is supposed to pause/stop)
                    m_rendererModel.RenderLock.Release();
                }
            }
            catch (Exception e)
            {
                Dispatcher.BeginInvoke(Error, e.Message);
            }
            Core.mufflon_destroy();
        }


        /// <summary>
        /// asynch thread: openGL initialization (pixel format and wgl context)
        /// </summary>
        private void InitializeOpenGl()
        {
            m_deviceContext = User32.GetDC(m_hWnd);

            var pfd = new Gdi32.PIXELFORMATDESCRIPTOR();
            pfd.Init(
                24, // color bits
                0, // depth bits
                0 // stencil bits
            );

            var pixelFormat = Gdi32.ChoosePixelFormat(m_deviceContext, ref pfd);
            if (!Gdi32.SetPixelFormat(m_deviceContext, pixelFormat, ref pfd))
                throw new Win32Exception(Marshal.GetLastWin32Error());

            var renderingContext = OpenGl32.wglCreateContext(m_deviceContext);
            if (renderingContext == IntPtr.Zero)
                throw new Win32Exception(Marshal.GetLastWin32Error());

            if (!OpenGl32.wglMakeCurrent(m_deviceContext, renderingContext))
                throw new Win32Exception(Marshal.GetLastWin32Error());

            // dll call: initialize glad etc.
            m_logCallbackPointer = new Core.LogCallback(Logger.log);
            if (!Core.mufflon_initialize(m_logCallbackPointer))
                throw new Exception(Core.core_get_dll_error());
            if (!Loader.loader_set_logger(m_logCallbackPointer))
                throw new Exception(Core.core_get_dll_error());
        }

        /// <summary>
        /// asynch thread: calls the dll resize() function if the client area was resized
        /// </summary>
        private void HandleResize()
        {
            // viewport resize?
            int newWidth = m_viewport.Width;
            int newHeight = m_viewport.Height;
            int newOffsetX = m_viewport.OffsetX;
            int newOffsetY = m_viewport.OffsetY;

            if (m_renderWidth != newWidth || m_renderHeight != newHeight ||
                m_renderOffsetX != newOffsetX || m_renderOffsetY != newOffsetY)
            {
                m_renderWidth = newWidth;
                m_renderHeight = newHeight;
                m_renderOffsetX = newOffsetX;
                m_renderOffsetY = newOffsetY;
                if (!Core.resize(m_renderWidth, m_renderHeight, m_renderOffsetX, m_renderOffsetY))
                    throw new Exception(Core.core_get_dll_error());
            }
        }

        /// <summary>
        /// asynch thread: sends queued commands to the dll
        /// </summary>
        private void HandleCommands()
        {
            while (m_commandQueue.TryDequeue(out var command))
            {
                //Core.execute_command(command);
                // TODO: reintroduce?
            }
        }

        /// <summary>
        /// construction of the child window
        /// </summary>
        /// <param name="hwndParent">handle of parent window (border)</param>
        /// <returns></returns>
        protected override HandleRef BuildWindowCore(HandleRef hwndParent)
        {
            m_hWnd = User32.CreateWindowEx(
                0, // dwstyle
                "static", // class name
                "", // window name
                (int)(User32.WS_FLAGS.WS_CHILD | User32.WS_FLAGS.WS_VISIBLE), // style
                0, // x
                0, // y
                (int)m_parent.ActualWidth, // width
                (int)m_parent.ActualHeight, // height
                hwndParent.Handle, // parent handle
                IntPtr.Zero, // menu
                IntPtr.Zero, // hInstance
                0 // param
            );

            m_renderThread = new Thread(new ParameterizedThreadStart(Render));
            m_renderThread.Start();

            return new HandleRef(this, m_hWnd);
        }

        /// <summary>
        /// stops render thread and performs destruction of window
        /// </summary>
        /// <param name="hwnd"></param>
        protected override void DestroyWindowCore(HandleRef hwnd)
        {
            // stop render thread
            m_isRunning = false;
            m_rendererModel.RenderLock.Release();
            m_renderThread.Join();

            // destroy resources
            User32.DestroyWindow(hwnd.Handle);
        }
    }
}
