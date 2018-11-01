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
    /// Interaction logic for OrthoCameraView.xaml
    /// </summary>
    public partial class OrthoCameraView : UserControl
    {
        public OrthoCameraView(OrthoCameraViewModel orthoCameraViewModel)
        {
            InitializeComponent();
            DataContext = orthoCameraViewModel;
        }
    }
}
