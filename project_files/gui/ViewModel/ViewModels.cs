using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using gui.Annotations;
using gui.Model;
using gui.Model.Camera;
using gui.Model.Light;
using gui.Model.Material;
using gui.Utility;
using gui.ViewModel.Camera;
using gui.ViewModel.Light;
using gui.ViewModel.Material;

namespace gui.ViewModel
{
    /// <summary>
    /// class containing all static view models
    /// </summary>
    public class ViewModels
    {
        public ConsoleViewModel Console { get; }
        public ViewportViewModel Viewport { get; }
        public CamerasViewModel Cameras { get; }
        public LightsViewModel Lights { get; }
        public MaterialsViewModel Materials { get; }

        private readonly Models m_models;

        public ViewModels(MainWindow window)
        {
            // model initialization
            m_models = new Models(window);

            // view model initialization
            Console = new ConsoleViewModel(m_models);
            Viewport = new ViewportViewModel(m_models);
            Cameras = new CamerasViewModel(m_models);
            Lights = new LightsViewModel(m_models);
            Materials = new MaterialsViewModel(m_models);

            // test cameras
            m_models.Cameras.Models.Add(new PinholeCameraModel());
            m_models.Cameras.Models.Add(new FocusCameraModel());
            m_models.Cameras.Models.Add(new OrthoCameraModel());

            m_models.Lights.Models.Add(new PointLightModel());
            m_models.Lights.Models.Add(new DirectionalLightModel());
            m_models.Lights.Models.Add(new SpotLightModel());
            m_models.Lights.Models.Add(new EnvmapLightModel());
            m_models.Lights.Models.Add(new GoniometricLightModel());

            m_models.Materials.Models.Add(new LambertMaterialModel());
            m_models.Materials.Models.Add(new TorranceMaterialModel());
            m_models.Materials.Models.Add(new WalterMaterialModel());
            m_models.Materials.Models.Add(new EmissiveMaterialModel());
            m_models.Materials.Models.Add(new OrennayarMaterialModel());

            // command initialization
        }
    }
}
