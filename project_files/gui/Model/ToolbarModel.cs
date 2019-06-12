using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using gui.Annotations;

namespace gui.Model
{
    public class ToolbarModel : INotifyPropertyChanged
    {
        public enum SelectionTools
        {
            Nothing,
            Material,
            Light,
            Instance
        }

        private SelectionTools m_activeTool = SelectionTools.Nothing;

        public SelectionTools ActiveTool
        {
            get => m_activeTool;
            set
            {
                if (m_activeTool == value) return;
                m_activeTool = value;
                OnPropertyChanged(nameof(ActiveTool));
            }
        }

        private uint? m_iterations = null;
        public uint? Iterations
        {
            get => m_iterations;
            set
            {
                if (m_iterations == value) return;
                m_iterations = value;
                OnPropertyChanged(nameof(Iterations));
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
