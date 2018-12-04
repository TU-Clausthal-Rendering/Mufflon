using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model.Camera;
using gui.Model.Light;
using gui.Model.Material;
using gui.Utility;

namespace gui.Model
{
    /// <summary>
    /// class containing all static models
    /// </summary>
    public class Models
    {
        public AppModel App { get; }
        public RendererModel Renderer { get; }
        public ViewportModel Viewport { get; }

        public SceneModel Scene { get; }
        public SynchronizedModelList<CameraModel> Cameras { get; }
        public SynchronizedModelList<LightModel> Lights { get; }
        public SynchronizedModelList<MaterialModel> Materials { get; }

        public ToolbarModel Toolbar { get; }


        public Models(MainWindow window)
        {
            Viewport = new ViewportModel();
            Renderer = new RendererModel();
            App = new AppModel(window, Viewport, Renderer);
            Scene = new SceneModel();
            Cameras = new SynchronizedModelList<CameraModel>();
            Lights = new SynchronizedModelList<LightModel>();
            Materials = new SynchronizedModelList<MaterialModel>();
            Toolbar = new ToolbarModel();
        }
    }
}
