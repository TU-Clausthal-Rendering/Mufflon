﻿using gui.Dll;
using gui.Properties;
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
using System.Windows.Shapes;

namespace gui.View
{
    /// <summary>
    /// Interaction logic for Settings.xaml
    /// </summary>
    public partial class AppSettings : Window
    {
        public class Severity
        {
            public Core.Severity Level { get; set; }
            public string Name { get; set; }
        }
        public class ProfilingLevel
        {
            public Core.ProfilingLevel Level { get; set; }
            public string Name { get; set; }
        }

        public ObservableCollection<Severity> LogLevels { get; } = new ObservableCollection<Severity>
        {
            new Severity(){ Level = Core.Severity.PEDANTIC, Name = "Pedantic" },
            new Severity(){ Level = Core.Severity.INFO, Name = "Info" },
            new Severity(){ Level = Core.Severity.WARNING, Name = "Warning" },
            new Severity(){ Level = Core.Severity.ERROR, Name = "Error" },
            new Severity(){ Level = Core.Severity.FATAL_ERROR, Name = "Fatal error" }
        };
        public ObservableCollection<ProfilingLevel> CoreProfilerLevels { get; } = new ObservableCollection<ProfilingLevel>()
        {
            new ProfilingLevel(){ Level = Core.ProfilingLevel.ALL, Name = "All" },
            new ProfilingLevel(){ Level = Core.ProfilingLevel.HIGH, Name = "High" },
            new ProfilingLevel(){ Level = Core.ProfilingLevel.LOW, Name = "Low" },
            new ProfilingLevel(){ Level = Core.ProfilingLevel.OFF, Name = "Off" }
        };
        public ObservableCollection<ProfilingLevel> LoaderProfilerLevels { get; } = new ObservableCollection<ProfilingLevel>()
        {
            new ProfilingLevel(){ Level = Core.ProfilingLevel.ALL, Name = "All" },
            new ProfilingLevel(){ Level = Core.ProfilingLevel.HIGH, Name = "High" },
            new ProfilingLevel(){ Level = Core.ProfilingLevel.LOW, Name = "Low" },
            new ProfilingLevel(){ Level = Core.ProfilingLevel.OFF, Name = "Off" }
        };

        public Severity LogLevel { get; set; }
        public ProfilingLevel CoreProfilerLevel { get; set; }
        public ProfilingLevel LoaderProfilerLevel { get; set; }

        public AppSettings()
        {
            InitializeComponent();
            LogLevel = LogLevels[Settings.Default.LogLevel];
            CoreProfilerLevel = CoreProfilerLevels[Settings.Default.CoreProfileLevel];
            LoaderProfilerLevel = LoaderProfilerLevels[Settings.Default.LoaderProfileLevel];
            DataContext = this;
        }

        private void OkButtonClick(object sender, RoutedEventArgs args)
        {
            Close();
            // Apply changed settings
            Settings.Default.LogLevel = (int)LogLevel.Level;
            Settings.Default.CoreProfileLevel = (int)CoreProfilerLevel.Level;
            Settings.Default.LoaderProfileLevel = (int)LoaderProfilerLevel.Level;

            Logger.LogLevel = LogLevel.Level; // Also sets the level in the DLLs
            if (CoreProfilerLevel.Level == Core.ProfilingLevel.OFF)
                Core.profiling_disable();
            else if (!Core.profiling_set_level(CoreProfilerLevel.Level))
                throw new Exception(Core.core_get_dll_error());
            if (LoaderProfilerLevel.Level == Core.ProfilingLevel.OFF)
                Loader.loader_profiling_disable();
            else if (!Loader.loader_profiling_set_level(LoaderProfilerLevel.Level))
                throw new Exception(Loader.loader_get_dll_error());
        }

        private void CancelButtonClick(object sender, RoutedEventArgs args)
        {
            Close();
            // Do not save settings
        }
    }
}
