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
using System.Windows.Input;

namespace gui.View
{
    // TODO remove logic from this class and put it into view model
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

        private readonly Models m_models;
        private ViewModels m_viewModels;

        // General settings
        public Severity LogLevel { get; set; }
        public ProfilingLevel CoreProfilerLevel { get; set; }
        public ProfilingLevel LoaderProfilerLevel { get; set; }
        // TODO: pattern file type checking?
        public string ScreenshotNamePattern { get; set; } = Settings.Default.ScreenshotNamePattern;
        public string ScreenshotFolder => Settings.Default.ScreenshotFolder;
        public StringCollection ScreenshotNamePatternHistory { get => Settings.Default.ScreenshotNamePatternHistory; }

        // Key bindings/gestures
        public string PlayPauseGesture { get; set; }
        public string ResetGesture { get; set; }
        public string ScreenshotGesture { get; set; }
        public string CameraMoveToggleGesture { get; set; }

        public AppSettings(ViewModels viewModels, Models models)
        {
            InitializeComponent();
            m_viewModels = viewModels;
            m_models = models;
            if (Settings.Default.ScreenshotNamePatternHistory == null)
                Settings.Default.ScreenshotNamePatternHistory = new StringCollection();
            if (Settings.Default.ScreenshotNamePatternHistory.Count == 0)
                Settings.Default.ScreenshotNamePatternHistory.Add(ScreenshotNamePattern);
            LogLevel = LogLevels[Settings.Default.LogLevel];
            CoreProfilerLevel = CoreProfilerLevels[Settings.Default.CoreProfileLevel];
            LoaderProfilerLevel = LoaderProfilerLevels[Settings.Default.LoaderProfileLevel];
            PlayPauseGesture = m_models.Settings.PlayPauseGestureString;
            ResetGesture = m_models.Settings.ResetGestureString;
            ScreenshotGesture = m_models.Settings.ScreenshotGestureString;
            CameraMoveToggleGesture = m_models.Settings.ToggleCameraMovementGestureString;

            DataContext = this;
            this.PreviewKeyDown += OnKeyPressed;
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
                Settings.Default.ScreenshotNamePatternHistory.Insert(0, ScreenshotNamePattern);
                if (Settings.Default.ScreenshotNamePatternHistory.Count > MAX_SCREENSHOT_PATTERN_HISTORY)
                    Settings.Default.ScreenshotNamePattern.Remove(Settings.Default.ScreenshotNamePatternHistory.Count - 1);
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

           
            m_models.Settings.PlayPauseGestureString = PlayPauseGesture;
            m_models.Settings.ResetGestureString = ResetGesture;
            m_models.Settings.ScreenshotGestureString = ScreenshotGesture;
            m_models.Settings.ToggleCameraMovementGestureString = CameraMoveToggleGesture;

            m_models.Settings.Save();
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
                    Settings.Default.ScreenshotFolder = dialog.SelectedPath;
                    OnPropertyChanged(nameof(ScreenshotFolder));
                }
            }
        }

        private void OnKeyPressed(object sender, System.Windows.Input.KeyEventArgs args)
        {
            if(args.Key == System.Windows.Input.Key.Escape)
                Close();
        }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
