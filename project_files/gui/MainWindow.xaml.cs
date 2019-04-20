using System;
using System.Windows;
using gui.Dll;
using gui.Model;
using gui.ViewModel;

namespace gui
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private Models m_models;
        private ViewModels m_viewModels;

        public MainWindow()
        {
            InitializeComponent();

            if (!Core.mufflon_initialize())
                throw new Exception(Core.core_get_dll_error());

            m_models = new Models(this);
            m_viewModels = new ViewModels(m_models);
            DataContext = m_viewModels;
        }

        public void GlHostOnError(string message)
        {
            MessageBox.Show(this, message, "OpenGL Thread Error");
            //Close();
        }

        protected override void OnClosed(EventArgs e)
        {
            m_models.Dispose();

            base.OnClosed(e);
        }
    }
}
