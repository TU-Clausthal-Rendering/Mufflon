using gui.Annotations;
using gui.Dll;
using gui.Model;
using gui.Properties;
using gui.ViewModel;
using System;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Forms;

namespace gui.View
{
    /// <summary>
    /// Interaction logic for Settings.xaml
    /// </summary>
    public partial class AppSettings : Window, INotifyPropertyChanged
    {
        private static readonly int MAX_SCREENSHOT_PATTERN_HISTORY = 10;
        public static readonly string ScreenshotNamePatternTooltip = "Specifies the pattern by which screenshots taken will be named\n" +
            "Valid tags are:\n" +
            "\t#scene - the name of the scene\n" +
            "\t#renderer - the name of the active renderer" +
            "\t#scenario - the name of the active scenario" +
            "\t#iteration - the iteration at which the screenshot was taken";

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

        private ViewModels m_viewModels;

        // General settings
        private string m_screenshotFolder;
        public Severity LogLevel { get; set; }
        public ProfilingLevel CoreProfilerLevel { get; set; }
        public ProfilingLevel LoaderProfilerLevel { get; set; }
        // TODO: pattern file type checking?
        public string ScreenshotNamePattern { get; set; } = Settings.Default.ScreenshotNamePattern;
        public string ScreenshotFolder { get => m_screenshotFolder; }
        public StringCollection ScreenshotNamePatternHistory { get => Settings.Default.ScreenShotNamePatternHistory; }

        // Key bindings/gestures
        public string PlayPauseGesture { get; set; }
        public string ResetGesture { get; set; }
        public string ScreenshotGesture { get; set; }

        public AppSettings(ViewModels viewModels)
        {
            InitializeComponent();
            m_viewModels = viewModels;
            if (Settings.Default.ScreenshotFolder == null || Settings.Default.ScreenshotFolder.Length == 0)
                Settings.Default.ScreenshotFolder = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            m_screenshotFolder = Settings.Default.ScreenshotFolder;
            if (Settings.Default.ScreenShotNamePatternHistory == null)
                Settings.Default.ScreenShotNamePatternHistory = new StringCollection();
            if (Settings.Default.ScreenShotNamePatternHistory.Count == 0)
                Settings.Default.ScreenShotNamePatternHistory.Add(ScreenshotNamePattern);
            LogLevel = LogLevels[Settings.Default.LogLevel];
            CoreProfilerLevel = CoreProfilerLevels[Settings.Default.CoreProfileLevel];
            LoaderProfilerLevel = LoaderProfilerLevels[Settings.Default.LoaderProfileLevel];
            PlayPauseGesture = viewModels.Toolbar.PlayPauseCommand.getCurrentGesture();
            ResetGesture = viewModels.Toolbar.ResetCommand.getCurrentGesture();
            ScreenshotGesture = viewModels.Toolbar.SaveScreenShotCommand.getCurrentGesture();

            OnPropertyChanged(nameof(ScreenshotFolder));
            OnPropertyChanged(nameof(ScreenshotNamePatternHistory));
            DataContext = this;
        }

        private void OkButtonClick(object sender, RoutedEventArgs args)
        {
            Close();
            // Apply changed settings
            Settings.Default.LogLevel = (int)LogLevel.Level;
            Settings.Default.CoreProfileLevel = (int)CoreProfilerLevel.Level;
            Settings.Default.LoaderProfileLevel = (int)LoaderProfilerLevel.Level;
            if(Settings.Default.ScreenshotNamePattern != ScreenshotNamePattern)
            {
                Settings.Default.ScreenshotNamePattern = ScreenshotNamePattern;
                Settings.Default.ScreenShotNamePatternHistory.Insert(0, ScreenshotNamePattern);
                if (Settings.Default.ScreenShotNamePatternHistory.Count > MAX_SCREENSHOT_PATTERN_HISTORY)
                    Settings.Default.ScreenshotNamePattern.Remove(Settings.Default.ScreenShotNamePatternHistory.Count - 1);
            }
            Logger.LogLevel = LogLevel.Level; // Also sets the level in the DLLs
            if (CoreProfilerLevel.Level == Core.ProfilingLevel.OFF)
                Core.profiling_disable();
            else if (!Core.profiling_set_level(CoreProfilerLevel.Level))
                throw new Exception(Core.core_get_dll_error());
            if (LoaderProfilerLevel.Level == Core.ProfilingLevel.OFF)
                Loader.loader_profiling_disable();
            else if (!Loader.loader_profiling_set_level(LoaderProfilerLevel.Level))
                throw new Exception(Loader.loader_get_dll_error());

            // Keybinds
            m_viewModels.Toolbar.PlayPauseCommand.updateGesture(PlayPauseGesture);
            m_viewModels.Toolbar.ResetCommand.updateGesture(ResetGesture);
            m_viewModels.Toolbar.SaveScreenShotCommand.updateGesture(ScreenshotGesture);
        }

        private void CancelButtonClick(object sender, RoutedEventArgs args)
        {
            Close();
            // Do not save settings
        }

        private void ScreenshotFolder_Click(object sender, RoutedEventArgs e)
        {
            using (FolderBrowserDialog dialog = new FolderBrowserDialog())
            {
                if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK && !string.IsNullOrWhiteSpace(dialog.SelectedPath))
                {
                    m_screenshotFolder = dialog.SelectedPath;
                    OnPropertyChanged(nameof(ScreenshotFolder));
                }
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
