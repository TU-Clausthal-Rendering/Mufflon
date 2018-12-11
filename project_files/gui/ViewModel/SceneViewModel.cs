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
                m_models.Scene.loadScene(Path);
            }

            public event EventHandler CanExecuteChanged
            {
                add { }
                remove { }
            }
        }

        private static readonly int MAX_LAST_SCENES = 10;
        private Models m_models;
        private ObservableCollection<SceneMenuItem> m_lastScenes;
        public ObservableCollection<SceneMenuItem> LastScenes { get => m_lastScenes; }
        public bool CanLoadLastScenes { get => m_lastScenes.Count > 0 && !m_models.Renderer.IsRendering; }

        public SceneViewModel(Models models)
        {
            m_models = models;
            m_lastScenes = new ObservableCollection<SceneMenuItem>();
            if (Settings.Default.LastScenes == null)
                Settings.Default.LastScenes = new System.Collections.Specialized.StringCollection();
            foreach (var path in Settings.Default.LastScenes)
                m_lastScenes.Add(new SceneMenuItem(m_models) { Filename = Path.GetFileName(path), Path = path });
            m_models.Scene.PropertyChanged += addNewScene;
            m_models.Renderer.PropertyChanged += renderStatusChanged;
        }


        private void addNewScene(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(SceneModel.FullPath):
                    {
                        string newScenePath = m_models.Scene.FullPath;
                        // Check if the scene is already present in the list
                        int index = Settings.Default.LastScenes.IndexOf(newScenePath);
                        if(index > 0)
                        {
                            // Present, but not first
                            SceneMenuItem first = m_lastScenes.First();
                            m_lastScenes[0] = m_lastScenes[index];
                            m_lastScenes[index] = first;
                            Settings.Default.LastScenes[0] = newScenePath;
                            Settings.Default.LastScenes[index] = first.Path;
                            OnPropertyChanged(nameof(LastScenes));
                            OnPropertyChanged(nameof(CanLoadLastScenes));
                        } else if(index < 0)
                        {
                            // Not present
                            if (m_lastScenes.Count >= MAX_LAST_SCENES)
                            {
                                m_lastScenes.RemoveAt(m_lastScenes.Count - 1);
                                Settings.Default.LastScenes.RemoveAt(m_lastScenes.Count - 1);
                            }
                            m_lastScenes.Insert(0, new SceneMenuItem(m_models)
                            {
                                Filename = m_models.Scene.Filename,
                                Path = newScenePath
                            });
                            Settings.Default.LastScenes.Insert(0, newScenePath);
                            OnPropertyChanged(nameof(LastScenes));
                            OnPropertyChanged(nameof(CanLoadLastScenes));
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
