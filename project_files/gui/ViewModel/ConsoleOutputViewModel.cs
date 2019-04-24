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
using gui.Dll;
using gui.Model;
using gui.View;
using Brush = System.Windows.Media.Brush;
using Brushes = System.Windows.Media.Brushes;

namespace gui.ViewModel
{
    /// <summary>
    /// View Model for the console output window
    /// </summary>
    public class ConsoleOutputViewModel : INotifyPropertyChanged
    {
        public ObservableCollection<TextBox> Output { get; } = new ObservableCollection<TextBox>();

        private readonly Models m_models;
        private int m_maxMessages;

        public ConsoleOutputViewModel(Models models)
        {
            m_models = models;
            m_maxMessages = models.Settings.MaxConsoleMessages;
            Logger.Log += GlHostOnLog;
            models.Settings.PropertyChanged += OnSettingsChanged;
        }

        private void GlHostOnLog(string message, Brush color)
        {
            AddText(message, color);
        }

        private void OnSettingsChanged(object sender, PropertyChangedEventArgs args)
        {
            if (sender == m_models.Settings && args.PropertyName == nameof(SettingsModel.MaxConsoleMessages))
                m_maxMessages = m_models.Settings.MaxConsoleMessages;
        }

        public void AddText(string text, Brush color)
        {
            // Remove earlier messages if stack is full
            if(m_maxMessages > 0 && Output.Count >= m_maxMessages)
                Output.RemoveAt(0);
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
