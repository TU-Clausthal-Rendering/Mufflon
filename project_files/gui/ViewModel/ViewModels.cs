﻿using System;
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
using gui.Controller;
using gui.Model;
using gui.Model.Camera;
using gui.Model.Light;
using gui.Model.Material;
using gui.Utility;
using gui.View;
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
        public ConsoleInputViewModel ConsoleInput { get; }

        public ViewportViewModel Viewport { get; }
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

        public CameraController CameraController { get; }

        public ICommand AddLightCommand { get; }
        public ICommand LoadSceneCommand { get; }
        public ICommand SaveSceneCommand { get; }
        public ICommand SelectRendererCommand { get; }
        public ICommand OpenSettingsCommand { get; }
        public ICommand AdjustGammaUp { get; }
        public ICommand AdjustGammaDown { get; }

        public Dictionary<Key, ICommand> Keybindings { get; } = new Dictionary<Key, ICommand>();

        private readonly Models m_models;

        public ViewModels(Models models)
        {
            m_models = models;

            // view model initialization
            ConsoleOutput = new ConsoleOutputViewModel(m_models);
            ConsoleInput = new ConsoleInputViewModel(m_models, m_models.App.Window.ConsoleInputBox, ConsoleOutput);
            Viewport = new ViewportViewModel(m_models);
            CameraController = new CameraController(m_models, m_models.App.Window.BorderHost);
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

            // command initialization
            AddLightCommand = new AddLightCommand(m_models);
            LoadSceneCommand = new LoadSceneCommand(m_models);
            SaveSceneCommand = new SaveSceneCommand(m_models);
            SelectRendererCommand = new SelectRendererCommand(m_models);
            OpenSettingsCommand = new OpenSettingsCommand(m_models);
            AdjustGammaUp = new AdjustGammaUpCommand(m_models);
            AdjustGammaDown = new AdjustGammaDownCommand(m_models);

            KeyGestures = new KeyGestureViewModel(models);

            // Add keybindings
            Keybindings.Add(Key.OemPlus, AdjustGammaUp);
            Keybindings.Add(Key.Add, AdjustGammaUp);
            Keybindings.Add(Key.OemMinus, AdjustGammaDown);
            Keybindings.Add(Key.Subtract, AdjustGammaDown);
        }
    }
}
