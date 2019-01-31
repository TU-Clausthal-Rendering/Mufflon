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
        private readonly Stack<Window> m_windowStack = new Stack<Window>();

        public MainWindow Window { get; }
        public OpenGLHost GlHost { get; }
        public Window TopmostWindow => m_windowStack.Peek();

        public AppModel(MainWindow window, ViewportModel viewport, RendererModel rendererModel,
            RenderTargetSelectionModel targetModel, SettingsModel settings)
        {
            Window = window;
            m_windowStack.Push(window);

            // init gl host
            GlHost = new OpenGLHost(window, viewport, rendererModel, targetModel, settings);
            GlHost.Error += window.GlHostOnError;
            window.Loaded += (sender, args) => window.BorderHost.Child = GlHost;
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
