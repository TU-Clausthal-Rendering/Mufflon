using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using gui.Annotations;
using gui.Model.Camera;
using gui.Model.Controller;
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
        // models

        public AppModel App { get; }
        public SettingsModel Settings { get; }

        public RendererModel Renderer { get; }
        public ViewportModel Viewport { get; }

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

        public Models(MainWindow window)
        {
            // init models first
            Settings = new SettingsModel();
            Viewport = new ViewportModel();
            Renderer = new RendererModel();
            RenderTargetSelection = new RenderTargetSelectionModel();
            App = new AppModel(window, Viewport, Renderer, RenderTargetSelection, Settings);
            Toolbar = new ToolbarModel();

            // init controller last
            ScenarioChangedController = new ScenarioChangedController(this);
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
