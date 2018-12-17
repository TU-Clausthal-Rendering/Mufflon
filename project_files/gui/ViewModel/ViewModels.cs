using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Annotations;
using gui.Command;
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

        public ToolbarViewModel Toolbar { get; }
        public StatusbarViewModel Statusbar { get; }
        public ProfilerViewModel Profiler { get; }

        public RendererViewModel Renderer { get; }
        public SceneViewModel Scene { get; }

        public ICommand AddLightCommand { get; }
        public ICommand AddMaterialCommand { get; }
        public ICommand AddCameraCommand { get; }
        public ICommand LoadSceneCommand { get; }

        public ICommand SelectRendererCommand { get; }

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

            Toolbar = new ToolbarViewModel(m_models);
            Statusbar = new StatusbarViewModel(m_models);

            Profiler = new ProfilerViewModel(window, m_models);

            Renderer = new RendererViewModel(window, m_models);
            Scene = new SceneViewModel(window, m_models);

            // command initialization
            AddLightCommand = new AddLightCommand(m_models);
            AddCameraCommand = new AddCameraCommand(m_models);
            AddMaterialCommand = new AddMaterialCommand(m_models);
            LoadSceneCommand = new LoadSceneCommand(m_models);
            SelectRendererCommand = new SelectRendererCommand(m_models);
        }
    }
}
