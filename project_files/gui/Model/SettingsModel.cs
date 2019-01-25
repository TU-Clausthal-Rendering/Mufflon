using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Annotations;
using gui.Dll;
using gui.Properties;

namespace gui.Model
{
    public class SettingsModel : INotifyPropertyChanged
    {
        private static int MaxLastWorlds { get; } = 10;
        private static int MaxScreenshotNamingPatterns { get; } = 10;
        private readonly KeyGestureConverter m_gestureConverter = new KeyGestureConverter();

        public SettingsModel()
        {
            SynchronizeStringCollectionWithObservable(Settings.Default.LastWorlds, LastWorlds);
            SynchronizeStringCollectionWithObservable(Settings.Default.ScreenshotNamePatternHistory, ScreenshotNamePatternHistory);
            if(ScreenshotNamePatternHistory.Count == 0)
                ScreenshotNamePatternHistory.Add(ScreenshotNamePattern);

            LoadGestures();
            SetLogLevel(LogLevel);

            if (string.IsNullOrEmpty(Settings.Default.ScreenshotFolder))
                Settings.Default.ScreenshotFolder = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);

            // forward setting changed events
            Settings.Default.PropertyChanged += AppSettingsOnPropertyChanged;
        }

        public void Save()
        {
            // store gestures
            Settings.Default.ScreenshotGesture = ScreenshotGestureString;
            Settings.Default.PlayPauseGesture = PlayPauseGestureString;
            Settings.Default.ResetGesture = ResetGestureString;
            Settings.Default.ToggleCameraMovementGesture = ToggleCameraMovementGestureString;

            // store collections
            Settings.Default.LastWorlds = ConvertToStringCollection(LastWorlds);
            Settings.Default.ScreenshotNamePatternHistory = ConvertToStringCollection(ScreenshotNamePatternHistory);

            // save settings
            Settings.Default.Save();
        }

        private void LoadGestures()
        {
            ScreenshotGestureString = Settings.Default.ScreenshotGesture;
            PlayPauseGestureString = Settings.Default.PlayPauseGesture;
            ResetGestureString = Settings.Default.ResetGesture;
            ToggleCameraMovementGestureString = Settings.Default.ToggleCameraMovementGesture;
        }

        private void SetLogLevel(Core.Severity severity)
        {
            if (!Core.core_set_log_level(severity))
                throw new Exception(Core.core_get_dll_error());
            if (!Loader.loader_set_log_level(severity))
                throw new Exception(Loader.loader_get_dll_error());
            // TODO redundant?
            Logger.LogLevel = LogLevel;
        }

        /// <summary>
        /// observable collection cannot be chosen in settings
        /// => save string collection as observable in SettingsModel
        /// </summary>
        private void SynchronizeStringCollectionWithObservable(StringCollection src, ObservableCollection<string> dest)
        {
            dest.Clear();
            if(src == null) return;

            foreach (var world in src)
            {
                dest.Add(world);
            }
        }

        private StringCollection ConvertToStringCollection(ObservableCollection<string> src)
        {
            var res = new StringCollection();
            foreach (var item in src)
            {
                res.Add(item);
            }

            return res;
        }


        private void AppSettingsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Settings.Default.LastWorldPath):
                    OnPropertyChanged(nameof(LastWorldPath));
                    break;
                case nameof(Settings.Default.LastSelectedRenderer):
                    OnPropertyChanged(nameof(LastSelectedRenderer));
                    break;
                case nameof(Settings.Default.LastWorlds):
                    // can be ignored
                    break;
                case nameof(Settings.Default.AutoStartOnLoad):
                    OnPropertyChanged(nameof(AutoStartOnLoad));
                    break;
                case nameof(Settings.Default.LogLevel):
                    SetLogLevel(LogLevel);
                    OnPropertyChanged(nameof(LogLevel));
                    break;
                case nameof(Settings.Default.CoreProfileLevel):
                    OnPropertyChanged(nameof(CoreProfileLevel));
                    if(CoreProfileLevel == Core.ProfilingLevel.Off)
                        Core.profiling_disable();
                    else if(!Core.profiling_set_level(CoreProfileLevel))
                        throw new Exception(Core.core_get_dll_error());
                    break;
                case nameof(Settings.Default.LoaderProfileLevel):
                    OnPropertyChanged(nameof(LoaderProfileLevel));
                    if(LoaderProfileLevel == Core.ProfilingLevel.Off)
                        Loader.loader_profiling_disable();
                    else if(!Loader.loader_profiling_set_level(LoaderProfileLevel))
                        throw new Exception(Loader.loader_get_dll_error());
                    break;
                case nameof(Settings.ScreenshotNamePattern):
                    OnPropertyChanged(nameof(ScreenshotNamePattern));
                    break;
                case nameof(Settings.ScreenshotFolder):
                    OnPropertyChanged(nameof(ScreenshotFolder));
                    break;
                case nameof(Settings.ScreenshotNamePatternHistory):
                    // can be ignored
                    break;
                case nameof(Settings.AllowCameraMovement):
                    OnPropertyChanged(nameof(AllowCameraMovement));
                    break;
            }
        }

        // TODO couple with last worlds?
        public string LastWorldPath
        {
            get => Settings.Default.LastWorldPath;
            set
            {
                Debug.Assert(value != null);
                Settings.Default.LastWorldPath = value;
            }
        }

        public int LastSelectedRenderer
        {
            get => Settings.Default.LastSelectedRenderer;
            set
            {
                Debug.Assert(value >= 0);
                Settings.Default.LastSelectedRenderer = value;
            }
        }

        public int LastSelectedRenderTarget
        {
            get => Settings.Default.LastSelectedRenderTarget;
            set
            {
                Debug.Assert(value >= 0);
                Settings.Default.LastSelectedRenderTarget = value;
            }
        }

        public LimitedCollection LastWorlds { get; } = new LimitedCollection(MaxLastWorlds);

        public bool AutoStartOnLoad
        {
            get => Settings.Default.AutoStartOnLoad;
            set => Settings.Default.AutoStartOnLoad = value;
        }

        // TODO use this
        public Core.Severity LogLevel
        {
            get => (Core.Severity) Settings.Default.LogLevel;
            set => Settings.Default.LogLevel = (int) value;
        }

        // TODO use this
        public Core.ProfilingLevel CoreProfileLevel
        {
            get => (Core.ProfilingLevel) Settings.Default.CoreProfileLevel;
            set => Settings.Default.CoreProfileLevel = (int) value;
        }

        // TODO use this
        public Core.ProfilingLevel LoaderProfileLevel
        {
            get => (Core.ProfilingLevel) Settings.Default.LoaderProfileLevel;
            set => Settings.Default.LoaderProfileLevel = (int) value;
        }

        // TODO use this
        public string ScreenshotNamePattern
        {
            get => Settings.Default.ScreenshotNamePattern;
            set => Settings.Default.ScreenshotNamePattern = value;
        }

        // TODO use this
        public string ScreenshotFolder
        {
            get => Settings.Default.ScreenshotFolder;
            set => Settings.Default.ScreenshotFolder = value;
        }

        // TODO use this
        public LimitedCollection ScreenshotNamePatternHistory { get; } = new LimitedCollection(MaxScreenshotNamingPatterns);

        private KeyGesture m_screenshotGesture;

        public KeyGesture ScreenshotGesture
        {
            get => m_screenshotGesture;
            set
            {
                if(m_screenshotGesture == value) return;
                m_screenshotGesture = value;
                OnPropertyChanged(nameof(ScreenshotGesture));
                OnPropertyChanged(ScreenshotGestureString);
            }
        }

        public string ScreenshotGestureString
        {
            get => m_screenshotGesture == null ? "" : m_gestureConverter.ConvertToString(m_screenshotGesture);
            set => ScreenshotGesture = (KeyGesture)m_gestureConverter.ConvertFromString(value);
        }

        private KeyGesture m_playPauseGesture;
        
        public KeyGesture PlayPauseGesture
        {
            get => m_playPauseGesture;
            set
            {
                if (m_playPauseGesture == value) return;
                m_playPauseGesture = value;
                OnPropertyChanged(nameof(PlayPauseGesture));
                OnPropertyChanged(nameof(PlayPauseGestureString));
            }
        }

        public string PlayPauseGestureString
        {
            get => m_playPauseGesture == null ? "" : m_gestureConverter.ConvertToString(m_playPauseGesture);
            set => PlayPauseGesture = (KeyGesture) m_gestureConverter.ConvertFromString(value);
        }

        private KeyGesture m_resetGesture;
        
        public KeyGesture ResetGesture
        {
            get => m_resetGesture;
            set
            {
                if(value == m_resetGesture) return;
                m_resetGesture = value;
                OnPropertyChanged(nameof(ResetGesture));
            }
        }

        public string ResetGestureString
        {
            get => m_resetGesture == null ? "" : m_gestureConverter.ConvertToString(m_resetGesture);
            set => ResetGesture = (KeyGesture)m_gestureConverter.ConvertFromString(value);
        }

        private KeyGesture m_toggleCameraMovementGesture;
        
        public KeyGesture ToggleCameraMovementGesture
        {
            get => m_toggleCameraMovementGesture;
            set
            {
                if(value == m_toggleCameraMovementGesture) return;
                m_toggleCameraMovementGesture = value;
                OnPropertyChanged(nameof(ToggleCameraMovementGesture));
            }
        }

        public string ToggleCameraMovementGestureString
        {
            get => m_toggleCameraMovementGesture == null
                ? ""
                : m_gestureConverter.ConvertToString(m_toggleCameraMovementGesture);
            set => ToggleCameraMovementGesture = (KeyGesture)m_gestureConverter.ConvertFromString(value);
        }

        public bool AllowCameraMovement
        {
            get => Settings.Default.AllowCameraMovement;
            set => Settings.Default.AllowCameraMovement = value;
        }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        /// <summary>
        /// modified collection that contains the maxItems strings
        /// </summary>
        public class LimitedCollection : ObservableCollection<string>
        {
            private readonly int m_maxItems;

            public LimitedCollection(int maxItems)
            {
                this.m_maxItems = maxItems;
            }

            protected override void InsertItem(int index, string item)
            {
                // only add if not already in collection
                if(Contains(item)) return;
                
                // this collection should not hold more than maxLastScenes scenes
                if (Count + 1 >= m_maxItems)
                {
                    // dont insert this item (would be the last in the list)
                    if(index == Count) return;
                    // remove the last item
                    RemoveAt(Count - 1);
                }

                base.InsertItem(index, item);
            }
        }
    }
}
