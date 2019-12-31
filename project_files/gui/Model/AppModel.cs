using System;
using System.Collections.Generic;
using System.Windows;
using gui.Dll;

namespace gui.Model
{
    /// <summary>
    /// relevant information about this instance
    /// </summary>
    public class AppModel
    {
        // Signal the rest of the app that we finished loading and may now
        // initialize things that were placeholders before properly (e.g. renderer count)
        public event EventHandler Loaded;

        private readonly Stack<Window> m_windowStack = new Stack<Window>();
        // this is required to prevent the callback from getting garbage collected
        private Core.LogCallback m_logCallbackPointer = null;

        public MainWindow Window { get; }
        public OpenGLHost GlHost { get; }
        public Window TopmostWindow => m_windowStack.Peek();

        public AppModel(MainWindow window)
        {
            Window = window;
            m_windowStack.Push(window);

            // Set the logger callback
            m_logCallbackPointer = new Core.LogCallback(Logger.log);

            // init gl host
            GlHost = new OpenGLHost(window);
            GlHost.Error += window.GlHostOnError;
            window.Loaded += OnWindowLoaded;
            //GlHost.InitializeOpenGl();

            // Now we may init the DLLs that we have an OpenGL context
            if (!Core.mufflon_initialize())
                throw new Exception(Core.core_get_dll_error());
            Core.profiling_enable();
            if (!Loader.loader_initialize())
                throw new Exception(Loader.loader_get_dll_error());
            Loader.loader_profiling_enable();
        }

        private void OnWindowLoaded(object sender, EventArgs args)
        {
            Window.RenderDisplay.BorderHost.Child = GlHost;
            if (!Core.core_set_logger(m_logCallbackPointer))
                throw new Exception(Core.core_get_dll_error());
            if (!Loader.loader_set_logger(m_logCallbackPointer))
                throw new Exception(Loader.loader_get_dll_error());
            GlHost.StartRenderLoop();
            Loaded(this, null);
        }

        /// <summary>
        /// shows a modal dialog and sets the correct dialog owner (topmost window)
        /// </summary>
        /// <param name="dialog"></param>
        /// <returns></returns>
        public bool? ShowDialog(Window dialog)
        {
            dialog.Owner = TopmostWindow;
            m_windowStack.Push(dialog);
            var res = dialog.ShowDialog();
            m_windowStack.Pop();
            return res;
        }
    }
}
