using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Input;
using gui.Annotations;
using gui.Model;

namespace gui.ViewModel
{
    /// <summary>
    /// View Model for the Console window
    /// </summary>
    public class ConsoleViewModel : INotifyPropertyChanged
    {
        public ObservableCollection<string> Output { get; } = new ObservableCollection<string>() { "line 1", "more code..." };

        private string m_input = "";
        public string Input
        {
            get => m_input;
            set
            {
                if (value == null || value == m_input) return;
                m_input = value;
                OnPropertyChanged(nameof(Input));
            }
        }

        private readonly Models m_models;

        public ConsoleViewModel(Models models)
        {
            m_models = models;
            m_models.App.Window.ConsoleInputBox.KeyDown += ConsoleInputBoxOnKeyDown;
            m_models.App.GlHost.Log += GlHostOnLog;
        }

        private void GlHostOnLog(string message)
        {
            Output.Add(message);
            m_models.App.Window.ConsoleScrollViewer.ScrollToBottom();
        }

        private void ConsoleInputBoxOnKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key != Key.Enter || Input.Length == 0) return;
            
            // use input as command
            Output.Add(Input);
            m_models.App.GlHost.QueueCommand(Input);
            Input = "";

            m_models.App.Window.ConsoleInputBox.Focus();
            m_models.App.Window.ConsoleScrollViewer.ScrollToBottom();
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
