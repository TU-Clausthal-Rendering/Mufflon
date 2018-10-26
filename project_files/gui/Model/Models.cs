using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace gui.Model
{
    /// <summary>
    /// class containing all static models
    /// </summary>
    public class Models
    {
        public AppModel App { get; }

        public Models(MainWindow window)
        {
            App = new AppModel(window);
        }
    }
}
