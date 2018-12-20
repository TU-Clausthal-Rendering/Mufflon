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
    /// Interaction logic for ScenarioLoadStatus.xaml
    /// </summary>
    public partial class ScenarioLoadStatus : Window
    {
        private TextBlock m_loadingText;

        public ScenarioLoadStatus(string scenarioName)
        {
            InitializeComponent();
            Owner = Application.Current.Windows.OfType<Window>().SingleOrDefault(x => x.IsActive);
            DataContext = this;
            m_loadingText = (TextBlock)FindName("LoadingTextBlock");
            m_loadingText.Text = "Loading '" + scenarioName + "'...";
            Show();
            Owner.IsEnabled = false;
        }

        protected override void OnClosed(EventArgs e)
        {
            base.OnClosed(e);
            Owner.IsEnabled = true;
        }
    }
}
