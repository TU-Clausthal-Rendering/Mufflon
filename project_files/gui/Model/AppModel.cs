using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace gui.Model
{
    /// <summary>
    /// relevant information about this instance
    /// </summary>
    public class AppModel
    {
        //public App App { get; }
        public MainWindow Window { get; }

        public AppModel(MainWindow window)
        {
            Window = window;
        }
    }
}
