using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using gui.Annotations;
using gui.Model;
using gui.Properties;

namespace gui.ViewModel
{
    public class SceneViewModel : INotifyPropertyChanged
    {
        public class SceneMenuItem : ICommand
        {
            private Models m_models;

            public string Filename { get; set; }
            public string Path { get; set; }
            public ICommand Command { get; }

            public SceneMenuItem(Models models)
            {
                m_models = models;
                Command = this;
            }

            SceneMenuItem(ICommand command)
            {
                Command = command;
            }

            public bool CanExecute(object parameter)
            {
                return !m_models.Renderer.IsRendering;
            }

            public void Execute(object parameter)
            {
                if(!m_models.Scene.loadScene(Path))
                {
                }
            }

            public event EventHandler CanExecuteChanged
            {
                add { }
                remove { }
            }
        }

        private Models m_models;
        private ObservableCollection<SceneMenuItem> m_lastScenes;
        public ObservableCollection<SceneMenuItem> LastScenes { get => m_lastScenes; }
        public bool CanLoadLastScenes { get => m_lastScenes.Count > 0 && !m_models.Renderer.IsRendering; }

        public SceneViewModel(Models models)
        {
            m_models = models;
            m_lastScenes = new ObservableCollection<SceneMenuItem>();
            foreach (string path in m_models.Scene.LastScenes)
            {
                m_lastScenes.Add(new SceneMenuItem(m_models)
                {
                    Filename = Path.GetFileName(path),
                    Path = path
                });
            }
            m_models.Scene.PropertyChanged += changeLastScenes;
            m_models.Renderer.PropertyChanged += renderStatusChanged;
        }

        private void changeLastScenes(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(SceneModel.LastScenes):
                    {
                        m_lastScenes.Clear();
                        foreach(string path in m_models.Scene.LastScenes)
                        {
                            m_lastScenes.Add(new SceneMenuItem(m_models)
                            {
                                Filename = Path.GetFileName(path),
                                Path = path
                            });
                        }
                        
                        break;
                    }
            }
            

        }

        private void renderStatusChanged(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(RendererModel.IsRendering):
                    OnPropertyChanged(nameof(CanLoadLastScenes));
                    break;
            }
        }

        #region PropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
}
