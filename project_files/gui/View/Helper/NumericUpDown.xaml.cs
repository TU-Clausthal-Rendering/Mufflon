using System;
using System.ComponentModel;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using gui.Annotations;

namespace gui.View.Helper
{
    /// <summary>
    /// Interaction logic for NumericUpDown.xaml
    /// </summary>
    ///
    public partial class NumericUpDown : UserControl, INotifyPropertyChanged
    {
        public int MinValue { get; set; } = Int32.MinValue;
        public int MaxValue { get; set; } = Int32.MaxValue;
        public int DefaultValue { get; set; } = 0;

        public static readonly DependencyProperty TextProperty =
            DependencyProperty.Register("Text", typeof(string),
                typeof(NumericUpDown), new PropertyMetadata(null));

        // TODO: doesn't update GUI
        public string Text
        {
            get { return NUDTextBox.Text; }
            set {
                if (NUDTextBox.Text == value) return;
                NUDTextBox.Text = value;
                OnPropertyChanged(nameof(Text));
                OnPropertyChanged(nameof(TextProperty));
            }
        }

        public NumericUpDown()
        {
            InitializeComponent();
            NUDTextBox.DataContext = this;
            Text = DefaultValue.ToString();
        }

        private void NUDButtonUp_Click(object sender, RoutedEventArgs e)
        {
            int number = 0;
            if (Text != "")
                number = Convert.ToInt32(Text);
            if (number < MaxValue)
                Text = (number + 1).ToString();
        }

        private void NUDButtonDown_Click(object sender, RoutedEventArgs e)
        {
            int number = 0;
            if (Text != "")
                number = Convert.ToInt32(Text);
            if (number > MinValue)
                Text = (number - 1).ToString();
        }

        private void NUDTextBox_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if(e.Key == Key.Up)
            {
                NUDButtonUp.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
                typeof(Button).GetMethod("set_IsPressed", BindingFlags.Instance | BindingFlags.NonPublic).Invoke(NUDButtonUp, new object[] { true });
            } else if (e.Key == Key.Down)
            {
                NUDButtonDown.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));
                typeof(Button).GetMethod("set_IsPressed", BindingFlags.Instance | BindingFlags.NonPublic).Invoke(NUDButtonDown, new object[] { true });
            }
        }

        private void NUDTextBox_PreviewKeyUp(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Up)
                typeof(Button).GetMethod("set_IsPressed", BindingFlags.Instance | BindingFlags.NonPublic).Invoke(NUDButtonUp, new object[] { false });

            if (e.Key == Key.Down)
                typeof(Button).GetMethod("set_IsPressed", BindingFlags.Instance | BindingFlags.NonPublic).Invoke(NUDButtonDown, new object[] { false });
        }

        private void NUDTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            int number = 0;
            if (Text != "")
                if (!int.TryParse(Text, out number))
                    Text = DefaultValue.ToString();
            if (number > MaxValue)
                Text = MaxValue.ToString();
            if (number < MinValue)
                Text = MinValue.ToString();
            NUDTextBox.SelectionStart = Text.Length;
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
