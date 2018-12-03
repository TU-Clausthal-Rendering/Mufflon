using System;
using gui.Dll;

namespace gui.Model
{
    /// <summary>
    /// relevant information about this instance
    /// </summary>
    public class AppModel
    {
        //public App App { get; }
        public MainWindow Window { get; }
        public OpenGLHost GlHost { get; }

        public AppModel(MainWindow window, ViewportModel viewport)
        {
            Window = window;

            // init gl host
            GlHost = new OpenGLHost(window.BorderHost, viewport);
            GlHost.Error += window.GlHostOnError;
            window.Loaded += (sender, args) => window.BorderHost.Child = GlHost;
        }
    }
}
