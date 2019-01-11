using System;
using System.IO;
using System.Windows.Forms;
using System.Windows.Input;
using System.ComponentModel;
using gui.Model;
using gui.Dll;
using gui.Properties;

namespace gui.Command
{
    class LoadSceneCommand : ICommand
    {
        private readonly Models m_models;
        private string m_lastDirectory;

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
                    m_models.Scene.LoadScene(file);
                    m_models.Renderer.Iteration = 0u;
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
