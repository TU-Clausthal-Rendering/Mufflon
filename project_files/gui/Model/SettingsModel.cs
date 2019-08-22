using System;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using System.Threading;
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
        private Timer m_saveTimer;

        public SettingsModel()
        {
            // Make sure the collections properly exist
            if (Settings.Default.LastWorlds == null)
                Settings.Default.LastWorlds = new StringCollection();
            if (Settings.Default.ScreenshotNamePatternHistory == null)
                Settings.Default.ScreenshotNamePatternHistory = new StringCollection();
            if (Settings.Default.RendererParameters == null)
                Settings.Default.RendererParameters = new StringCollection();

            // Create wrappers for the limited "history-like" collections
            LastWorlds = new LimitedStringCollection(Settings.Default.LastWorlds, MaxLastWorlds);
            ScreenshotNamePatternHistory = new LimitedStringCollection(Settings.Default.ScreenshotNamePatternHistory,
                MaxScreenshotNamingPatterns);
            RendererParameters = new LimitedStringCollection(Settings.Default.RendererParameters, int.MaxValue);

            LastWorlds.CollectionChanged += OnStringCollectionChanged;
            ScreenshotNamePatternHistory.CollectionChanged += OnStringCollectionChanged;
            RendererParameters.CollectionChanged += OnStringCollectionChanged;

            // Screenshot history has to have at least one entry
            if (ScreenshotNamePatternHistory.Empty)
                ScreenshotNamePatternHistory.PushFront(ScreenshotNamePattern);

            SetLogLevel(LogLevel);
            SetProfilerLevels();

            if (string.IsNullOrEmpty(ScreenshotFolder))
                ScreenshotFolder = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);

            // forward setting changed events
            m_saveTimer = new Timer(_ => {
                // TODO: idk why, but settings that weren't printed at least once are prone to not trigger
                // a save at all (seems buggy)
                Settings.Default.Save();
                m_saveTimer.Change(int.MaxValue, int.MaxValue);
            });
            Settings.Default.PropertyChanged += AppSettingsOnPropertyChanged;
        }

        public void Save()
        {
            // It appears that the settings object needs some time to make changes "visible"
            // Thus we use an invokable on a short timer that saves the settings with a slight delay.
            // Repeated "save" invokations simply reset the timer to avoid needless "double-saves".
            m_saveTimer.Change(200, int.MaxValue);
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
            if (!Core.profiling_set_level((Core.ProfilingLevel)CoreProfileLevel))
                throw new Exception(Core.core_get_dll_error());
            if (!Loader.loader_profiling_set_level((Core.ProfilingLevel)LoaderProfileLevel))
                throw new Exception(Loader.loader_get_dll_error());
        }

        private void OnStringCollectionChanged(object sender, NotifyCollectionChangedEventArgs args)
        {
            Save();
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
                case nameof(Settings.MaxConsoleMessages):
                    OnPropertyChanged(nameof(MaxConsoleMessages));
                    break;
            }

            Save();
        }

        public LimitedStringCollection LastWorlds { get; private set; }
        public LimitedStringCollection ScreenshotNamePatternHistory { get; private set; }

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

        public uint LastSelectedRenderer
        {
            get => Settings.Default.LastSelectedRenderer;
            set
            {
                Debug.Assert(value >= 0);
                Settings.Default.LastSelectedRenderer = value;
            }
        }

        public uint LastSelectedRendererVariation
        {
            get => Settings.Default.LastSelectedRendererVariation;
            set
            {
                Debug.Assert(value >= 0);
                Settings.Default.LastSelectedRendererVariation = value;
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

        public int MaxConsoleMessages
        {
            get => Settings.Default.MaxConsoleMessages;
            set => Settings.Default.MaxConsoleMessages = value;
        }

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

        public KeyGesture ScreenshotGesture
        {
            get => ScreenshotGestureString == null ? null
                : (KeyGesture)m_gestureConverter.ConvertFromString(ScreenshotGestureString);
        }

        public string ScreenshotGestureString
        {
            get => Settings.Default.ScreenshotGesture;
            set
            {
                if (value == Settings.Default.ScreenshotGesture) return;
                Settings.Default.ScreenshotGesture = value;
                OnPropertyChanged(nameof(ScreenshotGestureString));
                OnPropertyChanged(nameof(ScreenshotGesture));
            }
        }
        
        public KeyGesture PlayPauseGesture
        {
            get => PlayPauseGestureString == null ? null
                : (KeyGesture)m_gestureConverter.ConvertFromString(PlayPauseGestureString);
        }

        public string PlayPauseGestureString
        {
            get => Settings.Default.PlayPauseGesture;
            set
            {
                if (value == Settings.Default.PlayPauseGesture) return;
                Settings.Default.PlayPauseGesture = value;
                OnPropertyChanged(nameof(PlayPauseGestureString));
                OnPropertyChanged(nameof(PlayPauseGesture));
            }
        }
        
        public KeyGesture ResetGesture
        {
            get => ResetGestureString == null ? null
                : (KeyGesture)m_gestureConverter.ConvertFromString(ResetGestureString);
        }

        public string ResetGestureString
        {
            get => Settings.Default.ResetGesture;
            set
            {
                if (value == Settings.Default.ResetGesture) return;
                Settings.Default.ResetGesture = value;
                OnPropertyChanged(nameof(ResetGestureString));
                OnPropertyChanged(nameof(ResetGesture));
            }
        }
        
        public KeyGesture ToggleCameraMovementGesture
        {
            get => ToggleCameraMovementGestureString == null ? null
                : (KeyGesture)m_gestureConverter.ConvertFromString(ToggleCameraMovementGestureString);
        }

        public string ToggleCameraMovementGestureString
        {
            get => Settings.Default.ToggleCameraMovementGesture;
            set
            {
                if (value == Settings.Default.ToggleCameraMovementGesture) return;
                Settings.Default.ToggleCameraMovementGesture = value;
                OnPropertyChanged(nameof(ToggleCameraMovementGestureString));
                OnPropertyChanged(nameof(ToggleCameraMovementGesture));
            }
        }

        public bool AllowCameraMovement
        {
            get => Settings.Default.AllowCameraMovement;
            set => Settings.Default.AllowCameraMovement = value;
        }

        public bool InvertCameraControls
        {
            get => Settings.Default.InvertCameraControls;
            set => Settings.Default.InvertCameraControls = value;
        }

        public uint LastNIterationCommand
        {
            get => Settings.Default.LastNIterationCommand;
            set => Settings.Default.LastNIterationCommand = value;
        }

        public LimitedStringCollection RendererParameters { get; private set; }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }


        // Wrapper class for a string collection with limited entries (and limited operations).
        // Takes a collection and limit as argument, when insert operations exceed the limit
        // the last elements will be removed from the collection.
        public class LimitedStringCollection
        {
            private StringCollection m_collection;
            private int m_limit;

            public event NotifyCollectionChangedEventHandler CollectionChanged;

            public LimitedStringCollection(StringCollection collection, int limit)
            {
                m_collection = collection;
                m_limit = limit;
                // Initial trimming
                while (m_collection.Count > m_limit)
                    m_collection.RemoveAt(m_limit);
            }

            public void Add(string val)
            {
                int newIndex = m_collection.Count;
                m_collection.Add(val);
                CollectionChanged(this, new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Add, val, newIndex));
                PruneSingle();
            }

            public void PushFront(string val)
            {
                m_collection.Insert(0, val);
                CollectionChanged(this, new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Add, val, 0));
                PruneSingle();
            }

            public void RemoveAt(int index)
            {
                if (index >= Count)
                    throw new ArgumentOutOfRangeException();
                var item = m_collection[index];
                m_collection.RemoveAt(index);
                CollectionChanged(this, new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Remove, item, index));
            }

            public void Clear()
            {
                m_collection.Clear();
                CollectionChanged(this, new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Reset));
            }

            public int Count { get => m_collection.Count; }
            public bool Empty { get => Count == 0; }

            public int IndexOf(string val)
            {
                return m_collection.IndexOf(val);
            }

            public bool Contains(string val)
            {
                return m_collection.Contains(val);
            }

            public string this[int key]
            {
                get => m_collection[key];
                set
                {
                    if (key >= Count)
                        throw new ArgumentOutOfRangeException();
                    var oldItem = m_collection[key];
                    m_collection[key] = value;
                    CollectionChanged(this, new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Replace, value, oldItem, key));
                }
            }

            public StringEnumerator GetEnumerator()
            {
                return m_collection.GetEnumerator();
            }

            private void PruneSingle()
            {
                if(m_collection.Count > m_limit)
                {
                    var removed = m_collection[m_limit];
                       m_collection.RemoveAt(m_limit);
                    CollectionChanged(this, new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Remove, removed, m_limit));
                }
            }
        }
    }
}
