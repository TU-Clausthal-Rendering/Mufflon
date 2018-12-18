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

namespace gui.ViewModel
{
    /// <summary>
    /// View Model for a console input box
    /// </summary>
    public class ConsoleInputViewModel : INotifyPropertyChanged
    {
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
        private readonly TextBox m_inputBox;
        private readonly ConsoleOutputViewModel m_output;

        public ConsoleInputViewModel(Models models, TextBox inputBox, ConsoleOutputViewModel output)
        {
            m_models = models;
            m_inputBox = inputBox;
            m_output = output;
            m_inputBox.KeyDown += ConsoleInputBoxOnKeyDown;
        }

        private void ConsoleInputBoxOnKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key != Key.Enter || Input.Length == 0) return;

            // use input as command
            m_output.AddText(Input, Brushes.White);
            m_models.App.GlHost.QueueCommand(Input);
            Input = "";

            m_inputBox.Focus();
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
