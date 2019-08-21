using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using System.Threading;
using System.Windows.Forms.Design;
using gui.Annotations;
using gui.Dll;
using gui.Properties;
using Newtonsoft.Json;

namespace gui.Model
{
    // note: in order to save 
    public class SettingsModel : INotifyPropertyChanged
    {
        private static int MaxLastWorlds { get; } = 10;
        private static int MaxScreenshotNamingPatterns { get; } = 10;
        private readonly KeyGestureConverter m_gestureConverter = new KeyGestureConverter();

        public SettingsModel()
        {
            SetLogLevel(LogLevel);
            SetProfilerLevels();
        }

        public void Save()
        {
            var json = JsonConvert.SerializeObject(this, Formatting.Indented);
            File.WriteAllText("settings.json", json);
        }

        public static SettingsModel Load()
        {
            try
            {
                var json = File.ReadAllText("settings.json");
                var res = JsonConvert.DeserializeObject<SettingsModel>(json);
                return res;
            }
            catch (Exception) // use default settings
            {
                var res = new SettingsModel();
                // init screenshot name pattern history
                res.ScreenshotNamePatternHistory.Add(res.ScreenshotNamePattern);
                return res;
            }
        }

        private void SetLogLevel(Core.Severity severity)
        {
            if (!Core.core_set_log_level(severity))
                throw new Exception(Core.core_get_dll_error());
            if (!Loader.loader_set_log_level(severity))
                throw new Exception(Loader.loader_get_dll_error());
            Logger.LogLevel = LogLevel;
        }

        private void SetProfilerLevels()
        {
            if (!Core.profiling_set_level(CoreProfileLevel))
                throw new Exception(Core.core_get_dll_error());
            if (!Loader.loader_profiling_set_level(LoaderProfileLevel))
                throw new Exception(Loader.loader_get_dll_error());
        }

        // note: use AddWorld() to add items to this collection
        public ObservableCollection<string> LastWorlds { get; } = new ObservableCollection<string>();

        private static void InsertIntoLimitedUniqueCollection(ObservableCollection<string> collection, string item, int limit)
        {
            Debug.Assert(item != null);
            var index = collection.IndexOf(item);

            if (index > 0) // exists in list but not in the first spot
            {
                collection.RemoveAt(index);
                collection.Insert(0, item);
            }
            else if (index < 0) // does not exist yet
            {
                collection.Insert(0, item);
                while (collection.Count > limit) // limit collection
                {
                    collection.RemoveAt(collection.Count - 1);
                }
            }
        }

        public void AddWorld(string worldName)
        {
            InsertIntoLimitedUniqueCollection(LastWorlds, worldName, MaxLastWorlds);
        }

        // note: use AddScreenshotPattern to add items to this collection
        public ObservableCollection<string> ScreenshotNamePatternHistory { get; } = new ObservableCollection<string>();

        public void AddScreenshotPattern(string pattern)
        {
            InsertIntoLimitedUniqueCollection(ScreenshotNamePatternHistory, pattern, MaxScreenshotNamingPatterns);
        }

        private string m_lastWorldPath = "";
        public string LastWorldPath
        {
            get => m_lastWorldPath;
            set
            {
                Debug.Assert(value != null);
                if(value == m_lastWorldPath) return;
                m_lastWorldPath = value;
                OnPropertyChanged(nameof(LastWorldPath));
            }
        }

        private uint m_lastSelectedRenderer = 0;
        public uint LastSelectedRenderer
        {
            get => m_lastSelectedRenderer;
            set
            {
                if (value == m_lastSelectedRenderer) return;
                m_lastSelectedRenderer = value;
                OnPropertyChanged(nameof(LastSelectedRenderer));
            }
        }

        private uint m_lastSelectedRendererVariation = 0;
        public uint LastSelectedRendererVariation
        {
            get => m_lastSelectedRendererVariation;
            set
            {
                if(value == m_lastSelectedRendererVariation) return;
                m_lastSelectedRendererVariation = value;
                OnPropertyChanged(nameof(LastSelectedRendererVariation));
            }
        }

        private int m_lastSelectedRendererTarget = 0;
        public int LastSelectedRenderTarget
        {
            get => m_lastSelectedRendererTarget;
            set
            {
                Debug.Assert(value >= 0);
                if(value == m_lastSelectedRendererTarget) return;
                m_lastSelectedRendererTarget = value;
                OnPropertyChanged(nameof(LastSelectedRenderTarget));
            }
        }

        private int m_maxConsoleMessages = 50;
        public int MaxConsoleMessages
        {
            get => m_maxConsoleMessages;
            set
            {
                if(value == m_maxConsoleMessages) return;
                m_maxConsoleMessages = value;
                OnPropertyChanged(nameof(MaxConsoleMessages));
            }
        }

        private bool m_autoStartOnLoad = true;
        public bool AutoStartOnLoad
        {
            get => m_autoStartOnLoad;
            set
            {
                if(value == m_autoStartOnLoad) return;
                m_autoStartOnLoad = value;
                OnPropertyChanged(nameof(AutoStartOnLoad));
            }
        }

        // TODO use this
        private Core.Severity m_logLevel = Core.Severity.Pedantic;
        public Core.Severity LogLevel
        {
            get => m_logLevel;
            set
            {
                if(value == m_logLevel) return;
                m_logLevel = value;
                SetLogLevel(LogLevel);
                OnPropertyChanged(nameof(LogLevel));
            }
        }

        // TODO use this
        private Core.ProfilingLevel m_coreProfileLevel = Core.ProfilingLevel.All;
        public Core.ProfilingLevel CoreProfileLevel
        {
            get => m_coreProfileLevel;
            set
            {
                if(m_coreProfileLevel == value) return;
                m_coreProfileLevel = value;
                if (CoreProfileLevel == Core.ProfilingLevel.Off)
                    Core.profiling_disable();
                else if (!Core.profiling_set_level(CoreProfileLevel))
                    throw new Exception(Core.core_get_dll_error());
                OnPropertyChanged(nameof(CoreProfileLevel));
            }
        }

        // TODO use this
        private Core.ProfilingLevel m_loaderProfileLevel = Core.ProfilingLevel.All;
        public Core.ProfilingLevel LoaderProfileLevel
        {
            get => m_loaderProfileLevel;
            set
            {
                if(m_loaderProfileLevel == value) return;
                m_loaderProfileLevel = value;
                if (LoaderProfileLevel == Core.ProfilingLevel.Off)
                    Loader.loader_profiling_disable();
                else if (!Loader.loader_profiling_set_level(LoaderProfileLevel))
                    throw new Exception(Loader.loader_get_dll_error());
                OnPropertyChanged(nameof(LoaderProfileLevel));
            }
        }

        private string m_screenshotNamePattern = "#scene-#scenario-#renderer-#iteration";
        public string ScreenshotNamePattern
        {
            get => m_screenshotNamePattern;
            set
            {
                if(value == m_screenshotNamePattern) return;
                m_screenshotNamePattern = value;
                OnPropertyChanged(nameof(ScreenshotNamePattern));
                // add new pattern to the history
                AddScreenshotPattern(m_screenshotNamePattern);
            }
        }

        // TODO use this
        private string m_screenshotFolder = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
        public string ScreenshotFolder
        {
            get => m_screenshotFolder;
            set
            {
                if(value == m_screenshotFolder) return;
                m_screenshotFolder = value;
                OnPropertyChanged(nameof(ScreenshotFolder));
            }
        }

        [JsonIgnore]
        public KeyGesture ScreenshotGesture
        {
            get => ScreenshotGestureString == null ? null
                : (KeyGesture)m_gestureConverter.ConvertFromString(ScreenshotGestureString);
        }

        private string m_screenshotGestureString = "F2";
        public string ScreenshotGestureString
        {
            get => m_screenshotGestureString;
            set
            {
                if (value == m_screenshotGestureString) return;
                m_screenshotGestureString = value;
                OnPropertyChanged(nameof(ScreenshotGestureString));
                OnPropertyChanged(nameof(ScreenshotGesture));
            }
        }
        
        [JsonIgnore]
        public KeyGesture PlayPauseGesture
        {
            get => PlayPauseGestureString == null ? null
                : (KeyGesture)m_gestureConverter.ConvertFromString(PlayPauseGestureString);
        }

        private string m_playPauseGestureString = "ALT+P";
        public string PlayPauseGestureString
        {
            get => m_playPauseGestureString;
            set
            {
                if (value == m_playPauseGestureString) return;
                m_playPauseGestureString = value;
                OnPropertyChanged(nameof(PlayPauseGestureString));
                OnPropertyChanged(nameof(PlayPauseGesture));
            }
        }
        
        [JsonIgnore]
        public KeyGesture ResetGesture
        {
            get => ResetGestureString == null ? null
                : (KeyGesture)m_gestureConverter.ConvertFromString(ResetGestureString);
        }

        private string m_resetGestureString = "ALT+R";
        public string ResetGestureString
        {
            get => m_resetGestureString;
            set
            {
                if (value == m_resetGestureString) return;
                m_resetGestureString = value;
                OnPropertyChanged(nameof(ResetGestureString));
                OnPropertyChanged(nameof(ResetGesture));
            }
        }
        
        [JsonIgnore]
        public KeyGesture ToggleCameraMovementGesture
        {
            get => ToggleCameraMovementGestureString == null ? null
                : (KeyGesture)m_gestureConverter.ConvertFromString(ToggleCameraMovementGestureString);
        }

        private string m_toggleCameraMovementGestureString = "ALT+M";
        public string ToggleCameraMovementGestureString
        {
            get => m_toggleCameraMovementGestureString;
            set
            {
                if (value == m_toggleCameraMovementGestureString) return;
                m_toggleCameraMovementGestureString = value;
                OnPropertyChanged(nameof(ToggleCameraMovementGestureString));
                OnPropertyChanged(nameof(ToggleCameraMovementGesture));
            }
        }

        private bool m_allowCameraMovement = false;
        public bool AllowCameraMovement
        {
            get => m_allowCameraMovement;
            set
            {
                if(m_allowCameraMovement == value) return;
                m_allowCameraMovement = value;
                OnPropertyChanged(nameof(AllowCameraMovement));
            }
        }

        private bool m_invertCameraControls = false;
        public bool InvertCameraControls
        {
            get => m_invertCameraControls;
            set
            {
                if (value == m_invertCameraControls) return;
                m_invertCameraControls = value;
                OnPropertyChanged(nameof(InvertCameraControls));
            }
        }

        private uint m_lastNIterationCommand = 1000;
        public uint LastNIterationCommand
        {
            get => m_lastNIterationCommand;
            set
            {
                if(value == m_lastNIterationCommand) return;
                m_lastNIterationCommand = value;
                OnPropertyChanged(nameof(LastNIterationCommand));
            }
        }

        public ObservableCollection<string> RendererParameters { get; private set; } = new ObservableCollection<string>();

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            Save(); // TODO do we really want to save after each property change?

            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
