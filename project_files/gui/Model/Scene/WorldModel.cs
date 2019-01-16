using System;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using gui.Annotations;
using gui.Dll;
using gui.Properties;
using gui.Utility;

namespace gui.Model.Scene
{
    /// <summary>
    /// Additional information of the loaded scene (excluding light, materials and cameras)
    /// </summary>
    public class WorldModel : INotifyPropertyChanged
    {
        // path with filename and extension
        public string FullPath { get; }

        private readonly IntPtr m_handle;

        public ObservableCollection<ScenarioModel> Scenarios { get; } = new ObservableCollection<ScenarioModel>();

        private ScenarioModel m_currentScenario;
        public ScenarioModel CurrentScenario
        {
            get => m_currentScenario;
            set
            {
                Debug.Assert(value != null);
                Debug.Assert(Scenarios.Contains(m_currentScenario));

                if(ReferenceEquals(value, m_currentScenario)) return;
                // TODO add events for scenario started/finished loading (use ScenarioLoadStatus class)
                LoadScenarioAsync(value);
            }
        }

        public bool IsSane => Core.scene_is_sane();

        public void ReloadCurrentScenario()
        {
            if (Core.world_reload_current_scenario() == IntPtr.Zero)
                throw new Exception(Core.core_get_dll_error());
            OnPropertyChanged(nameof(IsSane));
        }

        private async void LoadScenarioAsync(ScenarioModel scenario)
        {
            var handle = scenario.Handle;

            await Task.Run(() =>
            {
                if (Core.world_load_scenario(handle) == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());
            });

            // set scenario properties
            m_currentScenario = scenario;
            OnPropertyChanged(nameof(CurrentScenario));
            OnPropertyChanged(nameof(BoundingBox));
        }

        public BoundingBox BoundingBox
        {
            get
            {
                if (!Core.scene_get_bounding_box(m_handle, out var min, out var max))
                    throw new Exception(Core.core_get_dll_error());

                return new BoundingBox(min.ToUtilityVec(), max.ToUtilityVec());
            }
        }

        // scene root directory
        public string Directory => Path.GetDirectoryName(FullPath);

        // filename with extension
        public string Filename => Path.GetFileName(FullPath);

        public WorldModel(IntPtr handle, string fullPath)
        {
            m_handle = handle;
            if(handle == IntPtr.Zero)
                throw new Exception("scene handle is nullptr");

            FullPath = fullPath;
            if (Settings.Default.LastScenes == null)
                Settings.Default.LastScenes = new StringCollection();

            // init scenarios
            uint count = Core.world_get_scenario_count();
            for (uint i = 0u; i < count; ++i)
                Scenarios.Add(new ScenarioModel(Core.world_get_scenario_by_index(i)));

            // which scenario was loaded?
            var loadedScenario = Core.world_get_current_scenario();
            foreach (var scenarioModel in Scenarios)
            {
                if (scenarioModel.Handle == loadedScenario)
                {
                    m_currentScenario = scenarioModel;
                    break;
                }
            }
            if(m_currentScenario == null)
                throw new Exception("could not find active scenario in loaded scenarios " + loadedScenario);
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
