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
using gui.ViewModel.Camera;

namespace gui.View.Camera
{
    /// <summary>
    /// Interaction logic for CameraView.xaml
    /// </summary>
    public partial class CameraView : UserControl
    {
        /// <summary>
        /// Creates the base camera view with the internal view
        /// </summary>
        /// <param name="dataContext">view model</param>
        /// <param name="internalView">camera type dependent view</param>
        public CameraView(CameraViewModel dataContext, UIElement internalView)
        {
            InitializeComponent();
            DataContext = dataContext;
            GroupItems.Children.Add(internalView);
        }
    }
}
