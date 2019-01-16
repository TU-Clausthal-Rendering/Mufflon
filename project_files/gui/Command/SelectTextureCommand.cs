using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Model;

namespace gui.Command
{
    public class SelectTextureCommand : ICommand
    {
        private readonly Models m_models;

        private readonly Func<string> m_getter;
        private readonly Action<string> m_setter;

        public SelectTextureCommand(Models models, Func<string> getter, Action<string> setter)
        {
            m_models = models;
            m_getter = getter;
            m_setter = setter;
        }

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public void Execute(object parameter)
        {
            var oldValue = m_getter.Invoke();
            if (System.IO.Path.IsPathRooted(oldValue))
            {
                // keep old value
                oldValue = oldValue.Replace('/', '\\');
            }
            else
            {
                // determine relative path
                var u = new Uri(m_models.World.Directory + "/" + oldValue);
                oldValue = u.AbsoluteUri;
                oldValue = oldValue.Replace('/', '\\');
            }

            var ofd = new Microsoft.Win32.OpenFileDialog
            {
                Multiselect = false,
                InitialDirectory = oldValue
            };

            if (ofd.ShowDialog(m_models.App.Window) != true) return;

            // convert absolute path to relative path
            var root = new Uri(m_models.World.FullPath);
            var newPath = new Uri(ofd.FileName);

            m_setter.Invoke(root.MakeRelativeUri(newPath).OriginalString);
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
