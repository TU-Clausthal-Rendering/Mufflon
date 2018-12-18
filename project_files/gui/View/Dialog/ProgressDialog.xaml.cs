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

namespace gui.View.Dialog
{
    /// <summary>
    /// Interaction logic for ProgressDialog.xaml
    /// Properties:
    /// WindowName - String
    /// ConsoleOutput - ConsoleOutputViewModel
    /// Progress - Value between 0 and 100
    /// </summary>
    public partial class ProgressDialog : Window
    {
        public ProgressDialog()
        {
            InitializeComponent();
        }
    }
}
