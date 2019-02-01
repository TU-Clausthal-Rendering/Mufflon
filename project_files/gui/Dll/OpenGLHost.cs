using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Timers;
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
        private readonly RenderTargetSelectionModel m_renderTargetModel;
        private readonly StatusbarModel m_statusbarModel;
        private readonly SettingsModel m_settings;

        // context creation
        private IntPtr m_hWnd = IntPtr.Zero;
        private IntPtr m_deviceContext = IntPtr.Zero;
        private IntPtr m_renderContext = IntPtr.Zero;

        // render thread (asynchronous openGL drawing)
        private Thread m_renderThread;
        private bool m_isRunning = true;

        // Blocker for when nothing is rendering to avoid busy-loop
        private readonly ConcurrentQueue<string> m_commandQueue = new ConcurrentQueue<string>();

        // helper to detect resize in render thread
        private int m_renderWidth = 0;
        private int m_renderHeight = 0;

        private Core.RenderTarget m_renderTarget;
        private bool m_varianceTarget;

        // TODO: this is only for testing purposes
        public static bool toggleRenderer = false;

        // Tracks whether the background was cleared in a window message
        private bool m_backgroundCleared = true;

        // this is required to prevent the callback from getting garbage collected
        private Core.LogCallback m_logCallbackPointer = null;

        public OpenGLHost(MainWindow window, ViewportModel viewport, RendererModel rendererModel,
            RenderTargetSelectionModel targetModel, StatusbarModel statusbar, SettingsModel settings)
        {
            m_window = window;
            m_parent = window.BorderHost;
            m_viewport = viewport;
            m_rendererModel = rendererModel;
            m_renderTargetModel = targetModel;
            m_statusbarModel = statusbar;
            m_settings = settings;
            m_window.MouseWheel += OnMouseWheel;
            m_window.SnapsToDevicePixels = true;
            // TODO: set the initial height here, but put a listener for window(!) size
            // TODO: On scenario change, also update the renderwith/height
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
                    m_statusbarModel.UpdateMemory();

                    //HandleCommands();
                    // Try to acquire the lock - if we're waiting, we're not rendering
                    m_rendererModel.RenderLock.WaitOne();
                    // Check if we've closed
                    if (!m_isRunning)
                        break;
                    HandleResize();

                    if (m_rendererModel.IsRendering)
                    {
                        if (m_settings.AllowCameraMovement)
                        {
                            // Check for keyboard input
                            float x = 0f;
                            float y = 0f;
                            float z = 0f;
                            // TODO: why does it need to be mirrored?
                            if (m_window.wasPressedAndClear(Key.W))
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

                            if (x != 0f || y != 0f || z != 0f)
                            {
                                if (!Core.scene_move_active_camera(x, y, z))
                                    throw new Exception(Core.core_get_dll_error());
                            }

                            // Check for mouse dragging
                            Vector drag = m_window.getMouseDiffAndReset();
                            if (drag.X != 0 || drag.Y != 0)
                            {
                                if (!Core.scene_rotate_active_camera(mouseSpeed * (float)drag.Y, -mouseSpeed * (float)drag.X, 0))
                                    throw new Exception(Core.core_get_dll_error());
                            }
                        }

                        if (!Core.render_iterate())
                            throw new Exception(Core.core_get_dll_error());

                        // We also let the GUI know that an iteration has taken place
                        Application.Current.Dispatcher.BeginInvoke(new Action(() => m_rendererModel.UpdateIterationCount()));
                    }

                    // Refresh the display
                    // Border always keeps aspect ratio
                    // Compute the offsets and dimensions of the viewport
                    int left = m_viewport.OffsetX;
                    int right = m_viewport.Width + m_viewport.OffsetX;
                    int invertedOffsetY = (m_viewport.DesiredHeight - m_viewport.Height) - m_viewport.OffsetY;
                    int bottom = invertedOffsetY;
                    int top = m_viewport.Height + invertedOffsetY;
                    UInt32 width = (UInt32)m_viewport.DesiredWidth;
                    UInt32 height = (UInt32)m_viewport.DesiredHeight;

                    IntPtr imageData = IntPtr.Zero;
                    if (!Core.core_get_target_image(m_renderTarget, m_varianceTarget, OpenGlDisplay.TextureFormat.Invalid,
                        false, out imageData))
                        throw new Exception(Core.core_get_dll_error());
                    if(imageData != IntPtr.Zero)
                        if (!OpenGlDisplay.opengldisplay_write(imageData))
                            throw new Exception(OpenGlDisplay.opengldisplay_get_dll_error());
                    if (!OpenGlDisplay.opengldisplay_display(left, right, bottom, top, width, height))
                        throw new Exception(OpenGlDisplay.opengldisplay_get_dll_error());
                    if (!Gdi32.SwapBuffers(m_deviceContext))
                        throw new Win32Exception(Marshal.GetLastWin32Error());

                    // We release it to give the GUI a chance to block us (ie. rendering is supposed to pause/stop)
                    m_rendererModel.RenderLock.Release();
                }
            }
            catch (Exception e)
            {
                Dispatcher.BeginInvoke(Error, e.Message);
            }

            // Clean up
            OpenGlDisplay.opengldisplay_destroy();
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

            // Create initial context for extension loading
            m_renderContext = OpenGl32.wglCreateContext(m_deviceContext);
            if(m_renderContext == IntPtr.Zero)
                throw new Win32Exception(Marshal.GetLastWin32Error());
            if (!OpenGl32.wglMakeCurrent(m_deviceContext, m_renderContext))
                throw new Win32Exception(Marshal.GetLastWin32Error());

            // Check if we can create the context with custom flags
            var wglCreateContextAttribsARB = OpenGl32.wglGetProcAddress<OpenGl32.WglCreateContextAttribsARB>("wglCreateContextAttribsARB");
            if(wglCreateContextAttribsARB != null)
            {
                int[] attribList = {
                    (int)OpenGl32.WglContextAttributeNames.CONTEXT_MAJOR_VERSION_ARB, 4,
                    (int)OpenGl32.WglContextAttributeNames.CONTEXT_PROFILE_MASK_ARB, (int) OpenGl32.WglContextProfileFlags.WGL_CONTEXT_CORE_PROFILE_BIT,
                    (int)OpenGl32.WglContextAttributeNames.CONTEXT_FLAGS_ARB, (int)OpenGl32.WglContextFlags.CONTEXT_FLAG_NO_ERROR_BIT
                };
                var extendedContext = wglCreateContextAttribsARB(m_deviceContext, IntPtr.Zero, attribList);
                if(extendedContext != IntPtr.Zero)
                {
                    if (!OpenGl32.wglMakeCurrent(m_deviceContext, extendedContext))
                        throw new Win32Exception(Marshal.GetLastWin32Error());
                    if (!OpenGl32.wglDeleteContext(m_renderContext))
                        throw new Win32Exception(Marshal.GetLastWin32Error());
                    m_renderContext = extendedContext;
                }
            }

            // dll call: initialize glad etc.
            if (!OpenGlDisplay.opengldisplay_initialize())
                throw new Exception(OpenGlDisplay.opengldisplay_get_dll_error());

            // Set the logger callback
            m_logCallbackPointer = new Core.LogCallback(Logger.log);
            if (!Core.mufflon_set_logger(m_logCallbackPointer))
                throw new Exception(Core.core_get_dll_error());
            if (!Loader.loader_set_logger(m_logCallbackPointer))
                throw new Exception(Core.core_get_dll_error());
            if (!OpenGlDisplay.opengldisplay_set_logger(m_logCallbackPointer))
                throw new Exception(OpenGlDisplay.opengldisplay_get_dll_error());

            // Update what renderers we might have
            Application.Current.Dispatcher.Invoke(new Action(() => m_rendererModel.RendererCount = Core.render_get_renderer_count()));
        }

        /// <summary>
        /// asynch thread: calls the dll resize() function if the client area was resized
        /// </summary>
        private void HandleResize()
        {
            // Check if we need to resize the screen texture
            int newWidth = m_viewport.RenderWidth;
            int newHeight = m_viewport.RenderHeight;
            Core.RenderTarget newTarget = m_renderTargetModel.VisibleTarget;
            bool newVarianceTarget = m_renderTargetModel.IsVarianceVisible;

            // TODO: disable the old target?
            if(newTarget != m_renderTarget || newVarianceTarget != m_varianceTarget)
            {
                if (!Core.render_is_render_target_enabled(m_renderTargetModel.VisibleTarget, m_renderTargetModel.IsVarianceVisible))
                {
                    // Disable previous render target
                    // TODO: better solution since we might want multiple render targets at a time
                    if(!Core.render_disable_render_target(m_renderTarget, m_varianceTarget ? 1u : 0u))
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.render_enable_render_target(m_renderTargetModel.VisibleTarget, m_renderTargetModel.IsVarianceVisible ? 1u : 0u))
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.render_reset())
                        throw new Exception(Core.core_get_dll_error());
                }
            }
            

            if (m_renderWidth != newWidth || m_renderHeight != newHeight || newTarget != m_renderTarget || newVarianceTarget != m_varianceTarget)
            {
                m_renderWidth = newWidth;
                m_renderHeight = newHeight;
                m_renderTarget = newTarget;
                m_varianceTarget = newVarianceTarget;

                // TODO: let GUI select what render target we render
                OpenGlDisplay.TextureFormat format;
                if(!Core.core_get_target_format(m_renderTarget, out format))
                    throw new Exception(Core.core_get_dll_error());
                if (!OpenGlDisplay.opengldisplay_resize_screen((UInt32)m_renderWidth, (UInt32)m_renderHeight, format))
                    throw new Exception(OpenGlDisplay.opengldisplay_get_dll_error());
            }
        }

        // Handle zooming when mouse wheel is used
        private void OnMouseWheel(object sender, MouseWheelEventArgs args)
        {
            float step = args.Delta < 0.0f ? 1.0f / 1.001f : 1.001f;
            float value = (float)Math.Pow(step, Math.Abs(args.Delta));

            float oldZoom = m_viewport.Zoom;
            int oldMaxOffsetX = m_viewport.DesiredWidth - Math.Min(m_viewport.Width, m_viewport.DesiredWidth);
            int oldMaxOffsetY = m_viewport.DesiredHeight - Math.Min(m_viewport.Height, m_viewport.DesiredHeight);
            m_viewport.Zoom *= value;

            // Zooming affects offset as well
            int newMaxOffsetX = m_viewport.DesiredWidth - Math.Min(m_viewport.Width, m_viewport.DesiredWidth);
            int newMaxOffsetY = m_viewport.DesiredHeight - Math.Min(m_viewport.Height, m_viewport.DesiredHeight);
            // Adjust the offset so that it stays roughly the same fractionally
            
            if (oldMaxOffsetX != 0)
                m_viewport.OffsetX = (int)(m_viewport.OffsetX * newMaxOffsetX / (float)oldMaxOffsetX);
            if (oldMaxOffsetY != 0)
                m_viewport.OffsetY = (int)(m_viewport.OffsetY * newMaxOffsetY / (float)oldMaxOffsetY);
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

        protected override IntPtr WndProc(IntPtr hwnd, int msg, IntPtr wParam, IntPtr lParam, ref bool handled)
        {
            if (msg == (int)Gdi32.WmMessages.ERASEBKGND)
            {
                m_backgroundCleared = true;
                handled = true;
            }
            if (msg == (int)Gdi32.WmMessages.PAINT && m_backgroundCleared && m_isRunning && !m_rendererModel.IsRendering)
            {
                m_backgroundCleared = false;
                // Iterate once to update the background
                m_rendererModel.Iterate(1u);
                handled = true;
            }
            return base.WndProc(hwnd, msg, wParam, lParam, ref handled);
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
