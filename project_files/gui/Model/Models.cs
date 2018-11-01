﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model.Camera;
using gui.Model.Light;
using gui.Utility;

namespace gui.Model
{
    /// <summary>
    /// class containing all static models
    /// </summary>
    public class Models
    {
        public AppModel App { get; }
        public ViewportModel Viewport { get; }
        public SynchronizedModelList<CameraModel> Cameras { get; }
        public SynchronizedModelList<LightModel> Lights { get; }

        public Models(MainWindow window)
        {
            Viewport = new ViewportModel();
            App = new AppModel(window, Viewport);
            Cameras = new SynchronizedModelList<CameraModel>();
            Lights = new SynchronizedModelList<LightModel>();
        }
    }
}
