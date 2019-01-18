using System;
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
    public class SaveSceneCommand : ICommand
    {
        private readonly Models m_models;
        private string m_lastDirectory;
        private KeyBinding m_keyBind;

        public SaveSceneCommand(Models models)
        {
            m_models = models;
            m_lastDirectory = Settings.Default.lastScenePath;
            if (m_lastDirectory.Length == 0)
                m_lastDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
            m_keyBind = new KeyBinding(this, new KeyGesture(Key.S, ModifierKeys.Control));
            System.Windows.Application.Current.MainWindow.InputBindings.Add(m_keyBind);
        }

        public bool CanExecute(object parameter)
        {
            return m_models.World != null; 
        }

        public virtual void Execute(object parameter)
        {
            using (SaveFileDialog dialog = new SaveFileDialog())
            {
                dialog.Title = "Select save location";
                dialog.Filter = "JSON files(*.json)|*.json";
                dialog.FilterIndex = 1;
                dialog.InitialDirectory = m_lastDirectory;
                dialog.AddExtension = false;
                dialog.CheckFileExists = false;
                dialog.CheckPathExists = true;
                dialog.RestoreDirectory = false;
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    string file = dialog.FileName;
                    m_lastDirectory = Path.GetDirectoryName(file);
                    SaveScene(file);
                }
            }
        }

        protected void SaveScene(string path)
        {

            var res = Loader.loader_save_scene(path);

            if (res == Loader.LoaderStatus.SUCCESS)
            {
                Logger.log("World '" + Path.GetFileName(path) + "' was saved successfully", Core.Severity.INFO);
            }
            else
            {
                System.Windows.MessageBox.Show("Failed to save scene!", "Error", MessageBoxButton.OK,
                    MessageBoxImage.Error);
                //Logger.log(e.Message, Core.Severity.FATAL_ERROR);
                return;
            }            
        }

        public event EventHandler CanExecuteChanged
        {
            add
            {
                CommandManager.RequerySuggested += value;
            }
            remove
            {
                CommandManager.RequerySuggested -= value;
            }
        }
    }
}
