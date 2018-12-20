using gui.Annotations;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
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
    /// Interaction logic for SceneLoadStatus.xaml
    /// </summary>
    public partial class SceneLoadStatus : Window, INotifyPropertyChanged
    {
        private Button m_cancelButton;
        private TextBlock m_loadingText;

        private bool m_canceled = false;
        public bool Canceled { get => m_canceled; }

        public SceneLoadStatus(string sceneName)
        {
            InitializeComponent();
            Owner = Application.Current.Windows.OfType<Window>().SingleOrDefault(x => x.IsActive);
            DataContext = this;
            m_cancelButton = (Button)FindName("CancelLoadButton");
            m_loadingText = (TextBlock)FindName("LoadingTextBlock");
            m_loadingText.Text = "Loading '" + sceneName + "'...";
            Show();
            Owner.IsEnabled = false;
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            m_canceled = true;
            // TODO: Keep the button visually pressed
            OnPropertyChanged(nameof(Canceled));
        }

        protected override void OnClosed(EventArgs e)
        {
            base.OnClosed(e);
            Owner.IsEnabled = true;
        }

        #region PropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
}
