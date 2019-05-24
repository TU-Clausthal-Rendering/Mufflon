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

        public event EventHandler Loop;
        public event EventHandler Destroy;

        // host of the HwndHost
        private readonly MainWindow m_window;

        // context creation
        private IntPtr m_hWnd = IntPtr.Zero;

        // render thread (asynchronous openGL drawing)
        private Thread m_renderThread;
        private bool m_isRunning;

        // Tracks when we're done initializing
        private bool m_renderThreadInitialized = false;
        // Wakes up a render thread that went to sleep after an error to perform cleanup
        private ManualResetEvent m_cleanup = new ManualResetEvent(true);

        // Indicates whether an OpenGL context has been created
        public bool OpenGlContextCreated { get; private set; } = false;

        public OpenGLHost(MainWindow window) {
            m_window = window;
        }

        /// <summary>
        /// asynch thread: render method
        /// </summary>
        private void Render() {
            m_isRunning = true;
            try {
                // Main loop
                while (m_isRunning)
                    Loop(this, null);
            } catch (Exception e) {
                m_cleanup.Reset();
                Dispatcher.BeginInvoke(Error, e.Message);
            }
        }

        /// <summary>
        /// asynch thread: openGL initialization (pixel format and wgl context)
        /// </summary>
        public void StartRenderLoop() {
            // Start our render loop which will have the OpenGL context bound
            m_renderThread = new Thread(new ThreadStart(InitializeOpenGl)) { Name = "RenderThread" };
            m_renderThread.Start();
            // wait until opengl was initialized on the render loop and mufflon renderers were loaded
            SpinWait.SpinUntil(() => m_renderThreadInitialized);
        }

        private void InitializeOpenGl() {
            IntPtr m_deviceContext = User32.GetDC(m_hWnd);

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
            IntPtr m_renderContext = OpenGl32.wglCreateContext(m_deviceContext);
            if (m_renderContext == IntPtr.Zero)
                throw new Win32Exception(Marshal.GetLastWin32Error());

            if (!OpenGl32.wglMakeCurrent(m_deviceContext, m_renderContext))
                throw new Win32Exception(Marshal.GetLastWin32Error());

            // Check if we can create the context with custom flags
            var wglCreateContextAttribsARB = OpenGl32.wglGetProcAddress<OpenGl32.WglCreateContextAttribsARB>("wglCreateContextAttribsARB");
            if(wglCreateContextAttribsARB != null) {
                int[] attribList = {
                    (int)OpenGl32.WglContextAttributeNames.CONTEXT_MAJOR_VERSION_ARB, 4,
                    (int)OpenGl32.WglContextAttributeNames.CONTEXT_PROFILE_MASK_ARB, (int) OpenGl32.WglContextProfileFlags.WGL_CONTEXT_CORE_PROFILE_BIT,
                    (int)OpenGl32.WglContextAttributeNames.CONTEXT_FLAGS_ARB, (int)OpenGl32.WglContextFlags.CONTEXT_FLAG_NO_ERROR_BIT
                };
                var extendedContext = wglCreateContextAttribsARB(m_deviceContext, IntPtr.Zero, attribList);
                if(extendedContext != IntPtr.Zero) {
                    if (!OpenGl32.wglMakeCurrent(m_deviceContext, extendedContext))
                        throw new Win32Exception(Marshal.GetLastWin32Error());
                    if (!OpenGl32.wglDeleteContext(m_renderContext))
                        throw new Win32Exception(Marshal.GetLastWin32Error());
                    m_renderContext = extendedContext;
                }
            }

            // Initialize the OpenGL display DLL
            OpenGlContextCreated = Core.mufflon_initialize_opengl();
            m_renderThreadInitialized = true;

            // Start the render loop
            Render();

            // Hold up with the cleanup until we destroy the window
            m_cleanup.WaitOne();
            if (OpenGlContextCreated)
                Core.mufflon_destroy_opengl();

            // Unload the core DLL
            Core.mufflon_destroy();
            
            // Release the contexts
            if (m_renderContext != IntPtr.Zero)
                User32.ReleaseDC(m_hWnd, m_renderContext);
            User32.ReleaseDC(m_hWnd, m_deviceContext);
        }
        
        protected override IntPtr WndProc(IntPtr hwnd, int msg, IntPtr wParam, IntPtr lParam, ref bool handled) {
            handled = false;
            return base.WndProc(hwnd, msg, wParam, lParam, ref handled);
        }

        /// <summary>
        /// construction of the child window
        /// </summary>
        /// <param name="hwndParent">handle of parent window (border)</param>
        /// <returns></returns>
        protected override HandleRef BuildWindowCore(HandleRef hwndParent) {
            m_hWnd = User32.CreateWindowEx(
                0, // dwstyle
                "static", // class name
                "", // window name
                (int)(User32.WS_FLAGS.WS_CHILD | User32.WS_FLAGS.WS_VISIBLE), // style
                0, // x
                0, // y
                (int)0, // width
                (int)0, // height
                hwndParent.Handle, // parent handle
                IntPtr.Zero, // menu
                IntPtr.Zero, // hInstance
                0 // param
            );

            if(m_hWnd == IntPtr.Zero)
                throw new Win32Exception(Marshal.GetLastWin32Error());


            return new HandleRef(this, m_hWnd);
        }

        /// <summary>
        /// stops render thread and performs destruction of window
        /// </summary>
        /// <param name="hwnd"></param>
        protected override void DestroyWindowCore(HandleRef hwnd) {
            m_isRunning = false;
            Destroy(this, null);
            m_cleanup.Set();
            m_renderThread.Join();

            // destroy resources
            User32.DestroyWindow(hwnd.Handle);
        }
    }
}
