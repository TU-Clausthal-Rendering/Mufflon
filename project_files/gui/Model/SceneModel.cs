using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using gui.Annotations;
using gui.Dll;
using gui.Properties;
using gui.View;

namespace gui.Model
{
    /// <summary>
    /// Additional information of the loaded scene (excluding light, materials and cameras)
    /// </summary>
    public class SceneModel : INotifyPropertyChanged
    {
        private static readonly int MAX_LAST_SCENES = 10;

        private string m_fullPath = null;
        private SceneLoadStatus m_cancelDialog;

        // path with filename and extension
        public string FullPath
        {
            get => m_fullPath;
            set
            {
                if (m_fullPath == value) return;
                m_fullPath = value;
                OnPropertyChanged(nameof(FullPath));
                OnPropertyChanged(nameof(IsLoaded));

                if (m_fullPath != null)
                {
                    // Check if we had this scene in the last X
                    // Check if the scene is already present in the list
                    int index = Settings.Default.LastScenes.IndexOf(m_fullPath);
                    if (index > 0)
                    {
                        // Present, but not first
                        string first = Settings.Default.LastScenes[0];
                        Settings.Default.LastScenes[0] = m_fullPath;
                        Settings.Default.LastScenes[index] = first;
                        OnPropertyChanged(nameof(LastScenes));
                    }
                    else if (index < 0)
                    {
                        // Not present
                        if (Settings.Default.LastScenes.Count >= MAX_LAST_SCENES)
                        {
                            Settings.Default.LastScenes.RemoveAt(Settings.Default.LastScenes.Count - 1);
                        }
                        Settings.Default.LastScenes.Insert(0, m_fullPath);
                        OnPropertyChanged(nameof(LastScenes));
                    }
                }
            }
        }

        public StringCollection LastScenes => Settings.Default.LastScenes;
        public ObservableCollection<string> Scenarios { get; }

        public bool IsLoaded
        {
            get => (m_fullPath != null);
        }

        // scene root directory
        public string Directory => Path.GetDirectoryName(FullPath);

        // filename with extension
        public string Filename => Path.GetFileName(FullPath);

        public SceneModel()
        {
            if (Settings.Default.LastScenes == null)
                Settings.Default.LastScenes = new StringCollection();
            Scenarios = new ObservableCollection<string>();
        }

        public void LoadScene(string path)
        {
            if (path == FullPath)
            {
                MessageBox.Show("Scene is already loaded", "", MessageBoxButton.OK,
                    MessageBoxImage.Exclamation);
                return; // true;
            }

            if (!File.Exists(path))
            {
                if (MessageBox.Show("Scene file '" + path + "' does not exists anymore; should it " +
                    "be removed from the list of recent scenes?", "Unable to load scene", MessageBoxButton.YesNo,
                    MessageBoxImage.Error) == MessageBoxResult.Yes)
                {
                    int index = Settings.Default.LastScenes.IndexOf(path);
                    if (index >= 0)
                    {
                        Settings.Default.LastScenes.RemoveAt(index);
                        OnPropertyChanged(nameof(LastScenes));
                    }
                }

                return;
            }

            m_cancelDialog = new SceneLoadStatus(Path.GetFileName(path));
            m_cancelDialog.PropertyChanged += cancelSceneLoad;
            LoadSceneAsynch(path);
        }

        private async void LoadSceneAsynch(string path)
        {
            var status = await Task.Run(() =>
            {
                var res = Loader.loader_load_json(path);
                return res;
            });
            m_cancelDialog.Close();

            if (status == Loader.LoaderStatus.ERROR)
            {
                MessageBox.Show("Failed to load scene!", "Error", MessageBoxButton.OK,
                    MessageBoxImage.Error);
                //Logger.log(e.Message, Core.Severity.FATAL_ERROR);
                FullPath = null;
                return;
            } else if(status == Loader.LoaderStatus.SUCCESS)
            {
                Logger.log("Scene '" + Path.GetFileName(path) + "' was loaded successfully", Core.Severity.INFO);

                // Set path and load scene properties
                FullPath = path;
                uint count = Core.world_get_scenario_count();
                Scenarios.Clear();
                for (uint i = 0u; i < count; ++i)
                    Scenarios.Add(Core.world_get_scenario_name_by_index(i));
                OnPropertyChanged(nameof(Scenarios));
            } else
            {
                Logger.log("Scene load was cancelled", Core.Severity.INFO);
            }
        }

        private void cancelSceneLoad(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(SceneLoadStatus.Canceled):
                    // Cancel the loading
                    // Ignore the return value, since cancelling isn't something really failable
                    if(m_cancelDialog.Canceled)
                        Loader.loader_abort();
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
