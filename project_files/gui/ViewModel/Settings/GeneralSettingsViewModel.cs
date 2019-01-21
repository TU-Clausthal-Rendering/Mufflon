using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Documents;
using System.Windows.Forms;
using System.Windows.Input;
using gui.Annotations;
using gui.Command;
using gui.Dll;
using gui.Model;
using gui.View.Helper;

namespace gui.ViewModel.Settings
{
    public class GeneralSettingsViewModel : INotifyPropertyChanged
    {
        private readonly Models m_models;

        public GeneralSettingsViewModel(Models models)
        {
            m_models = models;

            SelectScreenshotFolderCommand = new ActionCommand(ScreenshotFolderOnClick);
        }

        private void ScreenshotFolderOnClick()
        {
            using (FolderBrowserDialog dialog = new FolderBrowserDialog())
            {
                if (dialog.ShowDialog() != DialogResult.OK ||
                    string.IsNullOrWhiteSpace(dialog.SelectedPath)) return;

                ScreenshotFolder = dialog.SelectedPath;
                OnPropertyChanged(nameof(ScreenshotFolder));
            }
        }

        internal void LoadFromSettings()
        {
            var s = m_models.Settings;
            SelectedLogLevel = LogLevels.First(level => level.Cargo == s.LogLevel);
            SelectedCoreProfilerLevel = CoreProfilerLevels.First(level => level.Cargo == s.CoreProfileLevel);
            SelectedLoaderProfilerLevel = LoaderProfilerLevels.First(level => level.Cargo == s.LoaderProfileLevel);

            ScreenshotFolder = s.ScreenshotFolder;
            SelectedScreenshotNamePattern = s.ScreenshotNamePattern;
        }

        internal void StoreSettings()
        {
            var s = m_models.Settings;
            s.LogLevel = SelectedLogLevel.Cargo;
            s.CoreProfileLevel = SelectedCoreProfilerLevel.Cargo;
            s.LoaderProfileLevel = SelectedLoaderProfilerLevel.Cargo;
            s.ScreenshotNamePattern = SelectedScreenshotNamePattern;
            // TODO history?
        }

        public string ScreenshotNamePatternTooltip { get; } = 
            "Specifies the pattern by which screenshots taken will be named\n" +
            "Valid tags are:\n" +
            "\t#scene - the name of the scene\n" +
            "\t#renderer - the name of the active renderer" +
            "\t#scenario - the name of the active scenario" +
            "\t#iteration - the iteration at which the screenshot was taken";

        public ObservableCollection<EnumBoxItem<Core.Severity>> LogLevels { get; } = 
            EnumBoxItem<Core.Severity>.MakeCollection();

        public ObservableCollection<EnumBoxItem<Core.ProfilingLevel>> CoreProfilerLevels { get; } =
            EnumBoxItem<Core.ProfilingLevel>.MakeCollection();

        public ObservableCollection<EnumBoxItem<Core.ProfilingLevel>> LoaderProfilerLevels { get; } =
            EnumBoxItem<Core.ProfilingLevel>.MakeCollection();

        public EnumBoxItem<Core.Severity> SelectedLogLevel { get; set; }

        public EnumBoxItem<Core.ProfilingLevel> SelectedCoreProfilerLevel { get; set; }

        public EnumBoxItem<Core.ProfilingLevel> SelectedLoaderProfilerLevel { get; set; }

        public string ScreenshotFolder { get; set; }

        public string SelectedScreenshotNamePattern { get; set; }

        // the history should always be saved
        // TODO update this after selection change
        public ObservableCollection<string> ScreenshotNamePatterns => m_models.Settings.ScreenshotNamePatternHistory;

        public ICommand SelectScreenshotFolderCommand { get; }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
