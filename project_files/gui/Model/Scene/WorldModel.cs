using System;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using gui.Annotations;
using gui.Dll;
using gui.Model.Camera;
using gui.Model.Light;
using gui.Model.Material;
using gui.Properties;
using gui.Utility;

namespace gui.Model.Scene
{
    /// <summary>
    /// Additional information of the loaded scene (excluding light, materials and cameras)
    /// </summary>
    public sealed class WorldModel : INotifyPropertyChanged
    {
        // path with filename and extension
        public string FullPath { get; }

        private readonly IntPtr m_handle;

        public ObservableCollection<ScenarioModel> Scenarios { get; } = new ObservableCollection<ScenarioModel>();

        // TODO
        //public SynchronizedModelList<CameraModel> Cameras { get; } = new SynchronizedModelList<CameraModel>();

        // TODO make this readonly and add AddLight Method to WorldModel?
        public SynchronizedModelList<LightModel> Lights { get; } = new SynchronizedModelList<LightModel>();

        // TODO 
        //public SynchronizedModelList<MaterialModel> Materials { get; } = new SynchronizedModelList<MaterialModel>();

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
            if (Settings.Default.LastWorlds == null)
                Settings.Default.LastWorlds = new StringCollection();

            // first load lights and materials
            LoadLights();

            // load the scenario
            LoadScenarios();

            // notify lights etc. that the current scenario is loaded
            OnPropertyChanged(nameof(CurrentScenario));
        }

        private void LoadScenarios()
        {
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
            if (m_currentScenario == null)
                throw new Exception("could not find active scenario in loaded scenarios " + loadedScenario);
        }

        private void LoadLights()
        {
            var numPointLights = Core.world_get_point_light_count();
            for (var i = 0u; i < numPointLights; ++i)
                Lights.Models.Add(new PointLightModel(Core.world_get_light_handle(i, Core.LightType.POINT), this));

            var numSpotLights = Core.world_get_spot_light_count();
            for (var i = 0u; i < numSpotLights; ++i)
                Lights.Models.Add(new SpotLightModel(Core.world_get_light_handle(i, Core.LightType.SPOT), this));

            var numDirLights = Core.world_get_dir_light_count();
            for (var i = 0u; i < numDirLights; ++i)
                Lights.Models.Add(new DirectionalLightModel(Core.world_get_light_handle(i, Core.LightType.DIRECTIONAL), this));

            var numEnvLights = Core.world_get_env_light_count();
            for(var i = 0u; i < numEnvLights; ++i)
                Lights.Models.Add(new EnvmapLightModel(Core.world_get_light_handle(i, Core.LightType.ENVMAP), this));
        }

        #region PropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        private void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
}
