using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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

        public AppModel(MainWindow window)
        {
            Window = window;

            // init gl host
            GlHost = new OpenGLHost(window.BorderHost);
            GlHost.Error += window.GlHostOnError;
            window.Loaded += (sender, args) => window.BorderHost.Child = GlHost;
        }
    }
}
