using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace gui.Model
{
    /// <summary>
    /// Additional information of the loaded scene (excluding light, materials and cameras)
    /// </summary>
    public class SceneModel
    {
        // scene root directory
        public string Directory => System.IO.Path.GetDirectoryName(FullPath);

        // filename with extension
        public string Filename => System.IO.Path.GetFileName(FullPath);

        // path with filename and extension
        public string FullPath { get; set; } = "c:\\myScene\\json.txt";
    }
}
