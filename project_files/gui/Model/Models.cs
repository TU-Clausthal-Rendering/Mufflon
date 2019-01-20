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
        public SynchronizedModelList<CameraModel> Cameras { get; }
        public SynchronizedModelList<MaterialModel> Materials { get; }

        public ToolbarModel Toolbar { get; }

        public Models(MainWindow window)
        {
            Settings = new SettingsModel();
            Viewport = new ViewportModel();
            Renderer = new RendererModel();
            App = new AppModel(window, Viewport, Renderer, Settings);
            Cameras = new SynchronizedModelList<CameraModel>();
            Materials = new SynchronizedModelList<MaterialModel>();
            Toolbar = new ToolbarModel();
        }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
