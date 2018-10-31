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
using gui.Utility;
using gui.ViewModel.Camera;

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

        private readonly Models m_models;

        public ViewModels(MainWindow window)
        {
            // model initialization
            m_models = new Models(window);

            // view model initialization
            Console = new ConsoleViewModel(m_models);
            Viewport = new ViewportViewModel(m_models);
            Cameras = new CamerasViewModel(m_models);

            // test cameras
            m_models.Cameras.Models.Add(new PinholeCameraModel());
            m_models.Cameras.Models.Add(new PinholeCameraModel());
            m_models.Cameras.Models.Add(new PinholeCameraModel());

            // command initialization
        }
    }
}
