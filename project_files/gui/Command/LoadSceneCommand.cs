using System;
using System.IO;
using System.Windows.Forms;
using System.Windows.Input;
using gui.Model;
using gui.Dll;

namespace gui.Command
{
    class LoadSceneCommand : ICommand
    {
        private readonly Models m_models;
        private string m_lastDirectory;

        public LoadSceneCommand(Models models)
        {
            m_models = models;
            m_lastDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
        }

        public bool CanExecute(object parameter)
        {
            return true;
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
                    if (!Loader.loader_load_json(file))
                    {
                        MessageBox.Show("Failed to load scene!", "Error", MessageBoxButtons.OK,
                            MessageBoxIcon.Error);
                    }
                }
            }
            
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
