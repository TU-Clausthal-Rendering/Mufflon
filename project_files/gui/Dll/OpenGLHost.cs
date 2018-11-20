using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Interop;
using System.Windows.Media;
using gui.Model;

namespace gui.Dll
{
    public class OpenGLHost : HwndHost
    {
        // error callbacks (in case the openGL thread crashes)
        public delegate void ErrorEvent(string message);
        public event ErrorEvent Error;

        public delegate void LogEvent(string message, Brush color);
        public event LogEvent Log;

        // host of the HwndHost
        private readonly Border m_parent;
        // information about the viewport
        private readonly ViewportModel m_viewport;

        // context creation
        private IntPtr m_hWnd = IntPtr.Zero;
        private IntPtr m_deviceContext = IntPtr.Zero;

        // render thread (asynchronous openGL drawing)
        private Thread m_renderThread;
        private bool m_isRunning = true;
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

        public OpenGLHost(Border parent, ViewportModel viewport)
        {
            m_parent = parent;
            m_viewport = viewport;
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
            try
            {
                InitializeOpenGl();

                Core.RendererType rendererType = Core.RendererType.CPU_PT;
                IntPtr light = Core.world_add_point_light("testPointLight", new Core.Vec3(0, 0, 0),
                   new Core.Vec3(1, 1, 1));
                if(light == IntPtr.Zero)
                    throw new Exception(Core.GetDllError());
                IntPtr cam = Core.world_add_pinhole_camera("testPinholeCam", new Core.Vec3(0, 0, 0),
                    new Core.Vec3(0, 0, 1), new Core.Vec3(0, 1, 0), 1e-10f, 1e10f, 2.3f);
                if(cam == IntPtr.Zero)
                    throw new Exception(Core.GetDllError());
                IntPtr scenario = Core.world_create_scenario("testscenario");
                if(scenario == IntPtr.Zero)
                    throw new Exception(Core.GetDllError());
                if(!Core.scenario_add_light(scenario, "testPointLight"))
                    throw new Exception(Core.GetDllError());
                if(!Core.scenario_set_camera(scenario, cam))
                    throw new Exception(Core.GetDllError());
                if (!Core.scenario_set_resolution(scenario, 800, 600))
                    throw new Exception(Core.GetDllError());
                if (Core.world_load_scenario(scenario) == IntPtr.Zero)
                    throw new Exception(Core.GetDllError());
                if(!Core.render_enable_render_target(Core.RenderTarget.RADIANCE, 0))
                    throw new Exception(Core.GetDllError());
                if(!Core.render_enable_renderer(Core.RendererType.CPU_PT))
                    throw new Exception(Core.GetDllError());

                while (m_isRunning)
                {
                    HandleCommands();
                    HandleResize();

                    if(toggleRenderer)
                    {
                        if (rendererType == Core.RendererType.CPU_PT)
                            rendererType = Core.RendererType.GPU_PT;
                        else
                            rendererType = Core.RendererType.CPU_PT;

                        if (!Core.render_enable_renderer(rendererType))
                            throw new Exception(Core.GetDllError());
                        toggleRenderer = false;
                    }

                    if(!Core.render_iterate())
                      throw new Exception(Core.GetDllError());
                    if (!Core.display_screenshot())
                        throw new Exception(Core.GetDllError());
                    if(!Gdi32.SwapBuffers(m_deviceContext))
                        throw new Win32Exception(Marshal.GetLastWin32Error());
                    if(!Core.render_reset())
                        throw new Exception(Core.GetDllError());
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
            m_logCallbackPointer = new Core.LogCallback(LogCallback);
            if (!Core.mufflon_initialize(m_logCallbackPointer))
                throw new Exception(Core.GetDllError());
        }

        private void LogCallback(string message, int severity)
        {
            Dispatcher.BeginInvoke(Log, message, Brushes.White);
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
                    throw new Exception(Core.GetDllError());
            }
        }

        /// <summary>
        /// asynch thread: sends queued commands to the dll
        /// </summary>
        private void HandleCommands()
        {
            while (m_commandQueue.TryDequeue(out var command))
            {
                Core.execute_command(command);
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
            m_renderThread.Join();

            // destroy resources
            User32.DestroyWindow(hwnd.Handle);
        }
    }
}
