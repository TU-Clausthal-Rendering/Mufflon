using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace gui.View
{
    /// <summary>
    /// Interaction logic for IterateNDialog.xaml
    /// </summary>
    public partial class IterateNDialog : Window
    {
        public uint Iterations
        {
            get
            {
                uint val;
                if (uint.TryParse(IterationTextBox.Text, out val))
                    return val;
                return 0;
            }
        }

        public IterateNDialog(uint oldCount)
        {
            InitializeComponent();
            IterationTextBox.Text = oldCount.ToString();
            IterationTextBox.KeyDown += OnKeyDown;
            IterationTextBox.Focus();
        }

        private void OnKeyDown(object sender, KeyEventArgs args)
        {
            if (args.Key == Key.Enter)
                this.DialogResult = true;
        }

        private void OnOkClick(object sender, RoutedEventArgs args)
        {
            this.DialogResult = true;
        }

        private void OnCancelClick(object sender, RoutedEventArgs args)
        {
            this.DialogResult = false;
        }
    }
}
