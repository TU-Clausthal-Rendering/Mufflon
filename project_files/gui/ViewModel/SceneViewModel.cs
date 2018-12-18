using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Windows.Controls;
using System.Windows.Input;
using gui.Annotations;
using gui.Dll;
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
                m_models.Scene.LoadScene(Path);
            }

            public event EventHandler CanExecuteChanged
            {
                add { }
                remove { }
            }
        }

        private Models m_models;
        private ListBox m_scenarioBox;
        public ObservableCollection<SceneMenuItem> LastScenes { get; }
        public ObservableCollection<string> Scenarios { get => m_models.Scene.Scenarios; }
        public bool CanLoadLastScenes { get => LastScenes.Count > 0 && !m_models.Renderer.IsRendering; }

        public SceneViewModel(MainWindow window, Models models)
        {
            m_models = models;
            m_scenarioBox = (ListBox)window.FindName("ScenarioSelectionBox");
            m_scenarioBox.SelectionChanged += scenarioChanged;
            LastScenes = new ObservableCollection<SceneMenuItem>();
            foreach (string path in m_models.Scene.LastScenes)
            {
                LastScenes.Add(new SceneMenuItem(m_models)
                {
                    Filename = Path.GetFileName(path),
                    Path = path
                });
            }

            m_models.Scene.PropertyChanged += changeScene;
            m_models.Renderer.PropertyChanged += renderStatusChanged;
        }

        private void changeScene(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(SceneModel.LastScenes):
                    {
                        LastScenes.Clear();
                        foreach(string path in m_models.Scene.LastScenes)
                        {
                            LastScenes.Add(new SceneMenuItem(m_models)
                            {
                                Filename = Path.GetFileName(path),
                                Path = path
                            });
                        }
                        OnPropertyChanged(nameof(CanLoadLastScenes));
                        
                    }   break;
                case nameof(SceneModel.Scenarios):
                    {
                        IntPtr currentScenario = Core.world_get_current_scenario();
                        if (currentScenario == IntPtr.Zero)
                            throw new Exception(Core.core_get_dll_error());
                        string currentScenarioName = Core.world_get_scenario_name(currentScenario);
                        m_scenarioBox.SelectedItem = null;

                        // Select the loaded scenario (unsubscribe the event since that scenario is already loaded)
                        m_scenarioBox.SelectionChanged -= scenarioChanged;
                        foreach (var item in m_scenarioBox.Items)
                        {
                            if ((item as string) == currentScenarioName)
                            {
                                m_scenarioBox.SelectedItem = item;
                                break;
                            }
                        }
                        m_scenarioBox.SelectionChanged += scenarioChanged;
                    }   break;

            }
        }

        private void scenarioChanged(object sender, SelectionChangedEventArgs args)
        {
            if(args.AddedItems.Count > 0)
            {
                IntPtr scenario = Core.world_find_scenario(args.AddedItems[0] as string);
                if (scenario == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());
                if(Core.world_load_scenario(scenario) == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());
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
