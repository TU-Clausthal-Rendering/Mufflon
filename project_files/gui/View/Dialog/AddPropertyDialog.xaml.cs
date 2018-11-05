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
    /// Interaction logic for AddPropertyDialog.xaml.
    /// The add property dialog must be shown as a dialog because
    /// "add" and "cancel" set the dialog result property
    ///
    /// data context bindings:
    /// WindowTitle
    /// NameName - displayed name of the name property
    /// NameValue - value of the name property
    ///
    /// TypeName - displayed name of the type property
    /// TypeList - list of available items for types
    /// SelectedType - currently selected type
    /// </summary>
    public partial class AddPropertyDialog : Window
    {
        public AddPropertyDialog(object dataContext)
        {
            InitializeComponent();
            DataContext = dataContext;
        }

        private void AddOnClick(object sender, RoutedEventArgs e)
        {
            DialogResult = true;
            Close();
        }

        private void CancelOnClick(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
