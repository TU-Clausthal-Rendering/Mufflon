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
using System.Windows.Navigation;
using System.Windows.Shapes;
using gui.ViewModel.Material;

namespace gui.View.Material
{
    /// <summary>
    /// Interaction logic for RecursiveMaterialView.xaml
    /// </summary>
    public partial class RecursiveMaterialView : UserControl
    {
        /// <summary>
        /// Creates the base camera view with the internal view
        /// </summary>
        /// <param name="dataContext">view model</param>
        /// <param name="internalView">camera type dependent view</param>
        public RecursiveMaterialView(MaterialViewModel dataContext, UIElement internalView)
        {
            InitializeComponent();
            DataContext = dataContext;
            GroupItems.Children.Add(internalView);
        }

        private void OnSizeChanged(object sender, SizeChangedEventArgs e)
        {
            GroupHeaderGrid.Width = Math.Max(e.NewSize.Width - 20.0, 0.0);
        }
    }
}
