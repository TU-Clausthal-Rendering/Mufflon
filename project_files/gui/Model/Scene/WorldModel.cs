﻿using System;
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

        private readonly RendererModel m_rendererModel;

        public ObservableCollection<ScenarioModel> Scenarios { get; } = new ObservableCollection<ScenarioModel>();

        public LightsModel Lights { get; }

        public CamerasModel Cameras { get; }

        public MaterialsModel Materials { get; }

        /// <summary>
        /// references the previous scenario or null.
        /// This property is excluded from NotifyPropertyChanged since it changes with CurrentScenario
        /// </summary>
        public ScenarioModel PreviousScenario { get; private set; } = null;

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
            bool wasRendering = m_rendererModel.IsRendering;
            m_rendererModel.IsRendering = false;

            await Task.Run(() =>
            {
                if (Core.world_load_scenario(handle) == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());
            });

            // set scenario properties
            PreviousScenario = m_currentScenario;
            m_currentScenario = scenario;
            OnPropertyChanged(nameof(CurrentScenario));
            OnPropertyChanged(nameof(BoundingBox));
            OnPropertyChanged(nameof(IsSane));
            if (wasRendering)
                m_rendererModel.IsRendering = true;
        }

        public BoundingBox BoundingBox
        {
            get
            {
                if (!Core.scene_get_bounding_box(Core.world_get_current_scene(), out var min, out var max))
                    throw new Exception(Core.core_get_dll_error());

                return new BoundingBox(min.ToUtilityVec(), max.ToUtilityVec());
            }
        }

        // scene root directory
        public string Directory => Path.GetDirectoryName(FullPath);

        // filename with extension
        public string Filename => Path.GetFileName(FullPath);

        public WorldModel(RendererModel model, string fullPath)
        {
            m_rendererModel = model;
            FullPath = fullPath;

            // first load lights, cameras and materials
            Lights = new LightsModel();
            Materials = new MaterialsModel();
            Cameras = new CamerasModel();

            // load the scenario
            LoadScenarios();

            // notify lights etc. that the current scenario is loaded
            OnPropertyChanged(nameof(CurrentScenario));
        }

        private void LoadScenarios()
        {
            uint count = Core.world_get_scenario_count();
            for (uint i = 0u; i < count; ++i)
                Scenarios.Add(new ScenarioModel(this, Core.world_get_scenario_by_index(i)));

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
