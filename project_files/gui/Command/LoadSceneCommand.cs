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

namespace gui.Command
{
    public class LoadSceneCommand : ICommand
    {
        private readonly Models m_models;
        private string m_lastDirectory;
        private KeyBinding m_keyBind;

        private SceneLoadStatus m_cancelDialog;

        public LoadSceneCommand(Models models)
        {
            m_models = models;
            m_lastDirectory = models.Settings.LastWorldPath;
            if(m_lastDirectory.Length == 0)
                m_lastDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            m_keyBind = new KeyBinding(this, new KeyGesture(Key.O, ModifierKeys.Control));
            System.Windows.Application.Current.MainWindow.InputBindings.Add(m_keyBind);
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
                    m_models.Settings.LastWorldPath = m_lastDirectory;
                    LoadScene(file);
                }
            }
        }

        protected void LoadScene(string path)
        {
            if (!File.Exists(path))
            {
                if (System.Windows.MessageBox.Show("World file '" + path + "' does not exists anymore; should it " +
                                    "be removed from the list of recent scenes?", "Unable to load scene", MessageBoxButton.YesNo,
                        MessageBoxImage.Error) == MessageBoxResult.Yes)
                {
                    int index = m_models.Settings.LastWorlds.IndexOf(path);
                    if (index >= 0)
                    {
                        m_models.Settings.LastWorlds.RemoveAt(index);
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
                System.Windows.MessageBox.Show("Failed to load scene!", "Error", MessageBoxButton.OK,
                    MessageBoxImage.Error);
                //Logger.log(e.Message, Core.Severity.FATAL_ERROR);
                // remove old scene
                m_models.World = null;
                return;
            }
            else if (status == Loader.LoaderStatus.SUCCESS)
            {
                Logger.log("World '" + Path.GetFileName(path) + "' was loaded successfully", Core.Severity.Info);

                // Set path and load scene properties
                m_models.World = new WorldModel(m_models.Renderer, Core.world_get_current_scene(), path);
                RefreshLastScenes(path);
            }
            else
            {
                Logger.log("World load was cancelled", Core.Severity.Info);
            }
        }

        private void RefreshLastScenes(string path)
        {
            Debug.Assert(path != null);
            // Check if we had this scene in the last X
            // Check if the scene is already present in the list
            int index = m_models.Settings.LastWorlds.IndexOf(path);
            if (index > 0)
            {
                // Present, but not first
                m_models.Settings.LastWorlds.RemoveAt(index);
                m_models.Settings.LastWorlds.Insert(0, path);
            }
            else if (index < 0)
            {
                // Not present
                m_models.Settings.LastWorlds.Insert(0, path);
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
