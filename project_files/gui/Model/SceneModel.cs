using gui.Annotations;
using gui.Properties;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace gui.Model
{
    /// <summary>
    /// Additional information of the loaded scene (excluding light, materials and cameras)
    /// </summary>
    public class SceneModel : INotifyPropertyChanged
    {
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

            }
        }

        public bool IsLoaded
        {
            get => (m_fullPath != null);
        }

        // scene root directory
        public string Directory => System.IO.Path.GetDirectoryName(FullPath);

        // filename with extension
        public string Filename => System.IO.Path.GetFileName(FullPath);

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
