using System.Windows.Input;
using gui.Command;
using gui.Model;
using gui.ViewModel.Display;
using gui.ViewModel.Camera;
using gui.ViewModel.Dialog;
using gui.ViewModel.Light;
using gui.ViewModel.Material;

namespace gui.ViewModel
{
    /// <summary>
    /// class containing all static view models
    /// </summary>
    public class ViewModels
    {
        public ConsoleOutputViewModel ConsoleOutput { get; }

        public DisplayViewModel Display { get; }
        public CamerasViewModel Cameras { get; }
        public LightsViewModel Lights { get; }
        public MaterialsViewModel Materials { get; }

        public ToolbarViewModel Toolbar { get; }
        public StatusbarViewModel Statusbar { get; }
        public ProfilerViewModel Profiler { get; }

        public RendererViewModel Renderer { get; }
        public SceneViewModel Scene { get; }

        public RenderTargetSelectionViewModel RenderTargetSelection { get; }

        public KeyGestureViewModel KeyGestures { get; }

        public LoadWorldViewModel LoadWorld { get; }
        public AnimationFrameViewModel AnimationFrames { get; }

        public ICommand AddLightCommand { get; }
        public ICommand LoadSceneCommand { get; }
        public ICommand SaveSceneCommand { get; }
        public ICommand SelectRendererCommand { get; }
        public ICommand OpenSettingsCommand { get; }

        private readonly Models m_models;

        public ViewModels(Models models)
        {
            m_models = models;

            // view model initialization
            ConsoleOutput = new ConsoleOutputViewModel(m_models);
            Display = new DisplayViewModel(m_models);
            Cameras = new CamerasViewModel(m_models);
            Lights = new LightsViewModel(m_models);
            Materials = new MaterialsViewModel(m_models);

            Toolbar = new ToolbarViewModel(m_models);
            Statusbar = new StatusbarViewModel(m_models);

            Profiler = new ProfilerViewModel(m_models);

            Renderer = new RendererViewModel(m_models, Toolbar.PlayPauseCommand, Toolbar.ResetCommand);
            Scene = new SceneViewModel(m_models);

            RenderTargetSelection = new RenderTargetSelectionViewModel(m_models, Toolbar.ResetCommand);

            LoadWorld = new LoadWorldViewModel(m_models);
            AnimationFrames = new AnimationFrameViewModel(m_models);

            // command initialization
            AddLightCommand = new AddLightCommand(m_models);
            LoadSceneCommand = new LoadSceneCommand(m_models);
            SaveSceneCommand = new SaveSceneCommand(m_models);
            SelectRendererCommand = new SelectRendererCommand(m_models);
            OpenSettingsCommand = new OpenSettingsCommand(m_models);

            KeyGestures = new KeyGestureViewModel(models);
        }
    }
}
