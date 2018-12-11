using System.Collections.Generic;
using System.Collections.Specialized;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows;
using gui.Annotations;
using gui.Dll;
using gui.Properties;

namespace gui.Model
{
    /// <summary>
    /// Additional information of the loaded scene (excluding light, materials and cameras)
    /// </summary>
    public class SceneModel : INotifyPropertyChanged
    {
        private static readonly int MAX_LAST_SCENES = 10;

        private string m_fullPath = null;

        // path with filename and extension
        public string FullPath
        {
            get => m_fullPath;
            set
            {
                if (m_fullPath == value) return;
                m_fullPath = value;
                OnPropertyChanged(nameof(FullPath));

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

        public StringCollection LastScenes { get => Settings.Default.LastScenes; }

        public bool IsLoaded
        {
            get => (m_fullPath != null);
        }

        // scene root directory
        public string Directory => System.IO.Path.GetDirectoryName(FullPath);

        // filename with extension
        public string Filename => System.IO.Path.GetFileName(FullPath);

        public SceneModel()
        {
            if (Settings.Default.LastScenes == null)
                Settings.Default.LastScenes = new System.Collections.Specialized.StringCollection();
        }

        public bool loadScene(string path)
        {
            if(!File.Exists(path))
            {
                if(MessageBox.Show("Scene file '" + path + "' does not exists anymore; should it " +
                    "be removed from the list of recent scenes?", "Unable to load scene", MessageBoxButton.YesNo,
                    MessageBoxImage.Error) == MessageBoxResult.Yes)
                {
                    int index = Settings.Default.LastScenes.IndexOf(m_fullPath);
                    if(index >= 0)
                    {
                        Settings.Default.LastScenes.RemoveAt(index);
                        OnPropertyChanged(nameof(LastScenes));
                    }
                }
                return false;
            } else
            {
                if (!Loader.loader_load_json(path))
                {
                    MessageBox.Show("Failed to load scene!", "Error", MessageBoxButton.OK,
                        MessageBoxImage.Error);
                    return false;
                }
                else
                {
                    FullPath = path;
                    return true;
                }
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
