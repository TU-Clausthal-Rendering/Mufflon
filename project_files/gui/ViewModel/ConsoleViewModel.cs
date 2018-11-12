using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using gui.Annotations;
using gui.Model;
using Brush = System.Windows.Media.Brush;
using Brushes = System.Windows.Media.Brushes;

namespace gui.ViewModel
{
    /// <summary>
    /// View Model for the Console window
    /// </summary>
    public class ConsoleViewModel : INotifyPropertyChanged
    {
        public ObservableCollection<TextBox> Output { get; } = new ObservableCollection<TextBox>();

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

            //AddText("im white", Brushes.White);
            //AddText("im red", Brushes.Red);
        }

        private void GlHostOnLog(string message, Brush color)
        {
            AddText(message, color);
            m_models.App.Window.ConsoleScrollViewer.ScrollToBottom();
        }

        private void AddText(string text, Brush color)
        {
            Output.Add(new TextBox
            {
                Background = Brushes.Transparent,
                Foreground = color,
                IsReadOnly = true,
                TextWrapping = TextWrapping.Wrap,
                Text = text,
                BorderThickness = new Thickness(0.0),
                FontFamily = new FontFamily("Consolas")
            });
        }

        private void ConsoleInputBoxOnKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key != Key.Enter || Input.Length == 0) return;
            
            // use input as command
            AddText(Input, Brushes.White);
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
