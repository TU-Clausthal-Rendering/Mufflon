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
using gui.Dll;
using gui.ViewModel;

namespace gui
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        // Keeps track of the pressed status of keys
        private bool m_pressedW = false;
        private bool m_pressedS = false;
        private bool m_pressedD = false;
        private bool m_pressedA = false;
        private bool m_pressedSpace = false;
        private bool m_pressedLCtrl = false;
        // Stickies pressed keys until the next question
        private bool m_stickyW = false;
        private bool m_stickyS = false;
        private bool m_stickyD = false;
        private bool m_stickyA = false;
        private bool m_stickySpace = false;
        private bool m_stickyLCtrl = false;
        // To keep track of mouse dragging
        private Point m_lastMousePos;
        private Vector m_lastMouseDiff;

        public MainWindow()
        {
            InitializeComponent();

            DataContext = new ViewModels(this);
        }

        public void GlHostOnError(string message)
        {
            MessageBox.Show(this, message, "OpenGL Thread Error");
            //Close();
        }

        private void OnKeyDownHandler(object sender, KeyEventArgs e)
        {
            switch(e.Key)
            {
                case Key.W: m_pressedW = true; m_stickyW = true; break;
                case Key.S: m_pressedS = true; m_stickyS = true; break;
                case Key.D: m_pressedD = true; m_stickyD = true; break;
                case Key.A: m_pressedA = true; m_stickyA = true; break;
                case Key.Space: m_pressedSpace = true; m_stickySpace = true; break;
                case Key.LeftCtrl: m_pressedLCtrl = true; m_stickyLCtrl = true; break;
            }
        }

        private void OnKeyUpHandler(object sender, KeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.W: m_pressedW = false; break;
                case Key.S: m_pressedS = false; break;
                case Key.D: m_pressedD = false; break;
                case Key.A: m_pressedA = false; break;
                case Key.Space: m_pressedSpace = false; break;
                case Key.LeftCtrl: m_pressedLCtrl = false; break;
            }
        }

        public bool wasPressedAndClear(Key key)
        {
            switch (key)
            {
                case Key.W: { bool retval = m_pressedW || m_stickyW; m_stickyW = false; return retval; }
                case Key.S: { bool retval = m_pressedS || m_stickyS; m_stickyS = false; return retval; }
                case Key.D: { bool retval = m_pressedD || m_stickyD; m_stickyD = false; return retval; }
                case Key.A: { bool retval = m_pressedA || m_stickyA; m_stickyA = false; return retval; }
                case Key.Space: { bool retval = m_pressedSpace || m_stickySpace; m_stickySpace = false; return retval; }
                case Key.LeftCtrl: { bool retval = m_pressedLCtrl || m_stickyLCtrl; m_stickyLCtrl = false; return retval; }
            }
            return false;
        }

        private void OnMouseDownHandler(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                BorderHost.Focus();
                m_lastMousePos = e.MouseDevice.GetPosition(this);
            }
        }

        private void OnMouseMoveHandler(object sender, MouseEventArgs e)
        {
            if(e.LeftButton == MouseButtonState.Pressed)
            {
                Point currPos = e.MouseDevice.GetPosition(this);
                m_lastMouseDiff += currPos - m_lastMousePos;
                m_lastMousePos = currPos;
            }
        }

        public Vector getMouseDiffAndReset()
        {
            Vector last = m_lastMouseDiff;
            m_lastMouseDiff = new Vector();
            return last;
        }
    }
}
