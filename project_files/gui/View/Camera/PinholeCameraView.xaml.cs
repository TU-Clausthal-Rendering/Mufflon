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

namespace gui.View.Camera
{
    /// <summary>
    /// Interaction logic for PinholeCameraView.xaml
    /// </summary>
    public partial class PinholeCameraView : UserControl
    {
        public PinholeCameraView()
        {
            InitializeComponent();
        }

        public static readonly DependencyProperty CameraNameProperty = DependencyProperty.Register(
            "CameraName", typeof(string), typeof(PinholeCameraView), null           
        );

        public string CameraName
        {
            get => (string) GetValue(CameraNameProperty);
            set => SetValue(CameraNameProperty, value);
        }

        /*public static readonly DependencyProperty CameraFovProperty = DependencyProperty.Register(
            "CameraFov", typeof(float), typeof(PinholeCameraView), null
        );

        public float CameraFov
        {
            get => (float) GetValue(CameraFovProperty);
            set => SetValue(CameraFovProperty, value);
        }*/
    }
}
