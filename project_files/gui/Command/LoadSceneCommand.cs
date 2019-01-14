﻿using System;
using System.IO;
using System.Windows.Forms;
using System.Windows.Input;
using System.ComponentModel;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Windows;
using gui.Model;
using gui.Dll;
using gui.Model.Scene;
using gui.Properties;
using gui.View;
using MessageBox = System.Windows.MessageBox;

namespace gui.Command
{
    public class LoadSceneCommand : ICommand
    {
        private readonly Models m_models;
        private string m_lastDirectory;

        private SceneLoadStatus m_cancelDialog;
        private static readonly int MAX_LAST_SCENES = 10;

        public LoadSceneCommand(Models models)
        {
            m_models = models;
            m_lastDirectory = Settings.Default.lastScenePath;
            if(m_lastDirectory.Length == 0)
                m_lastDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
        }

        public bool CanExecute(object parameter)
        {
            return !m_models.Renderer.IsRendering;
        }

        public virtual void Execute(object parameter)
        {
            using(OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Title = "Select scene";
                dialog.Filter = "JSON files(*.json)|*.json";
                dialog.FilterIndex = 1;
                dialog.InitialDirectory = m_lastDirectory;
                dialog.AddExtension = true;
                dialog.CheckFileExists = true;
                dialog.CheckPathExists = true;
                dialog.Multiselect = false;
                dialog.RestoreDirectory = false;
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    string file = dialog.FileName;
                    m_lastDirectory = Path.GetDirectoryName(file);
                    Settings.Default.lastScenePath = m_lastDirectory;
                    LoadScene(file);
                }
            }
        }

        public void LoadScene(string path)
        {
            if (m_models.Scene != null && m_models.Scene.FullPath == path)
            {
                MessageBox.Show("Scene is already loaded", "", MessageBoxButton.OK,
                    MessageBoxImage.Exclamation);
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
                    }
                }

                return;
            }

            m_cancelDialog = new SceneLoadStatus(Path.GetFileName(path));
            m_cancelDialog.PropertyChanged += CancelSceneLoad;
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
                // remove old scene
                m_models.Scene = null;
                return;
            }
            else if (status == Loader.LoaderStatus.SUCCESS)
            {
                Logger.log("Scene '" + Path.GetFileName(path) + "' was loaded successfully", Core.Severity.INFO);

                // Set path and load scene properties
                m_models.Scene = new SceneModel(Core.world_get_current_scene(), path);
            }
            else
            {
                Logger.log("Scene load was cancelled", Core.Severity.INFO);
            }
        }

        private void RefreshLastScenes(string path)
        {
            Debug.Assert(path != null);
            // Check if we had this scene in the last X
            // Check if the scene is already present in the list
            int index = Settings.Default.LastScenes.IndexOf(path);
            if (index > 0)
            {
                // Present, but not first
                string first = Settings.Default.LastScenes[0];
                Settings.Default.LastScenes[0] = path;
                Settings.Default.LastScenes[index] = first;
            }
            else if (index < 0)
            {
                // Not present
                if (Settings.Default.LastScenes.Count >= MAX_LAST_SCENES)
                {
                    Settings.Default.LastScenes.RemoveAt(Settings.Default.LastScenes.Count - 1);
                }
                Settings.Default.LastScenes.Insert(0, path);
            }
        }

        private void CancelSceneLoad(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(SceneLoadStatus.Canceled):
                    // Cancel the loading
                    // Ignore the return value, since cancelling isn't something really failable
                    if (m_cancelDialog.Canceled)
                        Loader.loader_abort();
                    break;
            }
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
