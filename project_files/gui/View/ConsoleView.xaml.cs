using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
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

namespace gui.View
{
    /// <summary>
    /// Interaction logic for ConsoleView.xaml
    /// </summary>
    public partial class ConsoleView : UserControl
    {
        private int m_prevItemCount = 0;

        public ConsoleView()
        {
            InitializeComponent();
            
            ScrollViewer.ScrollChanged += ScrollViewerOnScrollChanged;
        }

        private void ScrollViewerOnScrollChanged(object sender, ScrollChangedEventArgs e)
        {
            var itemCount = Items.Items.Count;
            if (m_prevItemCount == itemCount) return;

            // a new item was added to the scroll viewer (scroll to bottom)
            // TODO: don't scroll to bottom if the user was scrolling (maybe look at verticalChange?)
            m_prevItemCount = itemCount;
            ScrollViewer.ScrollToBottom();
        }
    }
}
