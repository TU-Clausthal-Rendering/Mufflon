using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using gui.Properties;

namespace gui
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        private static readonly CultureInfo CultureInfo = new CultureInfo("en-US");

        public static CultureInfo GetCulture()
        {
            return CultureInfo;
        }

        void App_Exit(object sender, ExitEventArgs e)
        {
            Settings.Default.Save();
        }
    }
}
