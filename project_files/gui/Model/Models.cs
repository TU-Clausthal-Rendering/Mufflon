using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using gui.Annotations;
using gui.Controller.Renderer;
using gui.Dll;
using gui.Model.Camera;
using gui.Model.Controller;
using gui.Model.Display;
using gui.Model.Events;
using gui.Model.Light;
using gui.Model.Material;
using gui.Model.Scene;
using gui.Properties;
using gui.Utility;
using gui.View;

namespace gui.Model
{
    /// <summary>
    /// class containing all static models
    /// </summary>
    public class Models : INotifyPropertyChanged
    {
        // events
        public event LoadEventHandler OnWorldLoad;

        // models
        public AppModel App { get; }
        public SettingsModel Settings { get; }

        public RendererModel Renderer { get; }
        public RenderCameraModel RendererCamera { get; }
        public DisplayModel Display { get; }

        private WorldModel m_world = null;
        public WorldModel World
        {
            get => m_world;
            set
            {
                if(ReferenceEquals(value, m_world)) return;
                m_world = value;
                OnPropertyChanged(nameof(World));
            }
        }

        public ToolbarModel Toolbar { get; }

        public RenderTargetSelectionModel RenderTargetSelection { get; }

        // model controller 

        public ScenarioChangedController ScenarioChangedController { get; }
        public RendererCameraController RendererCameraController { get; }
        public RendererController RendererController{ get; }

        public StatusbarModel Statusbar { get; }

        public Models(MainWindow window) {
            // init models first
            App = new AppModel(window);
            Settings = new SettingsModel();
            Display = new DisplayModel();
            Renderer = new RendererModel();
            RendererCamera = new RenderCameraModel();
            RenderTargetSelection = new RenderTargetSelectionModel();
            Toolbar = new ToolbarModel();
            Statusbar = new StatusbarModel();

            // init controller last
            ScenarioChangedController = new ScenarioChangedController(this);
            RendererCameraController = new RendererCameraController(this, window.RenderDisplay);
            RendererController = new RendererController(this);
        }

        public async void LoadSceneAsynch(string path)
        {
            OnWorldLoad?.Invoke(this, new LoadEventArgs(LoadEventArgs.LoadStatus.Started, path));

            try
            {
                if (!File.Exists(path))
                    throw new Exception("File not found");

                // It is important that we clear the world from the renderer thread due to OpenGL allocations
                // We may assume that we're not rendering, so we can perform one iteration
                Renderer.ClearWorld();
                var status = await Task.Run(() =>
                {
                    var res = Loader.loader_load_json(path);
                    return res;
                });

                switch (status)
                {
                    case Loader.LoaderStatus.ERROR:
                        throw new Exception("Failed to load scene!");
                    case Loader.LoaderStatus.ABORT:
                        throw new Exception("World load was cancelled");
                    case Loader.LoaderStatus.SUCCESS:
                        World = new WorldModel(Renderer, path);
                        RefreshLastScenes(path);
                        break;
                }
            }
            catch (Exception e)
            {
                OnWorldLoad?.Invoke(this, new LoadEventArgs(LoadEventArgs.LoadStatus.Failed, e.Message));
                // remove old scene
                World = null;
                return;
            }
            OnWorldLoad?.Invoke(this, new LoadEventArgs(LoadEventArgs.LoadStatus.Finished));
        }

        public void CancelSceneLoad()
        {
            if(!Loader.loader_abort())
                throw new Exception(Core.core_get_dll_error());
        }

        private void RefreshLastScenes(string path)
        {
            Debug.Assert(path != null);
            // Check if we had this scene in the last X
            // Check if the scene is already present in the list
            int index = Settings.LastWorlds.IndexOf(path);
            if (index > 0)
            {
                // Present, but not first
                Settings.LastWorlds.RemoveAt(index);
                Settings.LastWorlds.PushFront(path);
            }
            else if (index < 0)
            {
                // Not present
                Settings.LastWorlds.PushFront(path);
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public void Dispose()
        {
            Settings.Save();
        }
    }
}
