using System;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using gui.Annotations;
using gui.Command;
using gui.Dll;
using gui.Model;
using gui.Model.Camera;
using gui.Model.Light;
using gui.Model.Material;
using gui.Model.Scene;
using gui.Properties;
using gui.Utility;
using gui.View;
using gui.View.Helper;

namespace gui.ViewModel
{
    public class SceneViewModel : INotifyPropertyChanged
    {
        public class SceneMenuItem : LoadSceneCommand
        {
            private Models m_models;

            public string Filename { get; set; }
            public string Path { get; set; }
            public ICommand Command { get; }

            public SceneMenuItem(Models models) : base(models)
            {
                m_models = models;
                Command = this;
            }

            public override void Execute(object parameter) => LoadScene(Path);
        }

        private Models m_models;
        public ObservableCollection<SceneMenuItem> LastScenes { get; } = new ObservableCollection<SceneMenuItem>();

        private ComboBoxItem<ScenarioModel> m_selectedScenario = null;
        public ComboBoxItem<ScenarioModel> SelectedScenario
        {
            get => m_selectedScenario;
            set
            {
                if(ReferenceEquals(value, m_selectedScenario)) return;
                m_selectedScenario = value;
                OnPropertyChanged(nameof(SelectedScenario));

                // load selected scenario
                if (m_selectedScenario == null) return;
                m_models.Scene.CurrentScenario = m_selectedScenario.Cargo;
            }
        }
        public ObservableCollection<ComboBoxItem<ScenarioModel>> Scenarios { get; } = new ObservableCollection<ComboBoxItem<ScenarioModel>>();
        public bool CanLoadLastScenes => LastScenes.Count > 0 && !m_models.Renderer.IsRendering;

        private ICommand m_playPause;
        private ICommand m_reset;
        public SceneViewModel(MainWindow window, Models models, ICommand playPause, ICommand reset)
        {
            m_models = models;
            m_playPause = playPause;
            m_reset = reset;

            if(Settings.Default.LastScenes == null)
                Settings.Default.LastScenes = new StringCollection();

            foreach (string path in Settings.Default.LastScenes)
            {
                LastScenes.Add(new SceneMenuItem(m_models)
                {
                    Filename = Path.GetFileName(path),
                    Path = path
                });
            }

            // Register the handle for scene changes
            m_models.Lights.Models.CollectionChanged += OnLightsCollectionChanged;
            m_models.Materials.Models.CollectionChanged += OnMaterialsCollectionChanged;
            m_models.Cameras.Models.CollectionChanged += OnCamerasCollectionChanged;

            // assume no scene loaded
            Debug.Assert(m_models.Scene == null);
            m_models.PropertyChanged += ModelsOnPropertyChanged;
            m_models.Renderer.PropertyChanged += OnRendererChange;

            Settings.Default.PropertyChanged += SettingsOnPropertyChanged;
        }

        private void ModelsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.Scene):
                    // subscribe to new model
                    if (m_models.Scene != null)
                    {
                        m_models.Scene.PropertyChanged += OnSceneChanged;
                        m_models.Scene.Scenarios.CollectionChanged += ScenariosOnCollectionChanged;
                        if (m_models.Scene.FullPath != null)
                        {
                            // Temporarily disable handlers
                            m_models.Lights.Models.CollectionChanged -= OnLightsCollectionChanged;
                            m_models.Materials.Models.CollectionChanged -= OnMaterialsCollectionChanged;
                            m_models.Cameras.Models.CollectionChanged -= OnCamerasCollectionChanged;
                            LoadSceneLights();
                            LoadSceneMaterials();
                            LoadSceneCameras();
                            m_models.Lights.Models.CollectionChanged += OnLightsCollectionChanged;
                            m_models.Materials.Models.CollectionChanged += OnMaterialsCollectionChanged;
                            m_models.Cameras.Models.CollectionChanged += OnCamerasCollectionChanged;
                        }
                    }

                    // refresh views
                    MakeScenerioViews();

                    break;
            }
        }

        private void SettingsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Settings.Default.LastScenes):
                    // TODO this wont be triggered => change to observable collection?
                    LastScenes.Clear();
                    foreach (string path in Settings.Default.LastScenes)
                    {
                        LastScenes.Add(new SceneMenuItem(m_models)
                        {
                            Filename = Path.GetFileName(path),
                            Path = path
                        });
                    }
                    OnPropertyChanged(nameof(CanLoadLastScenes));
                    break;
            }
        }

        private void OnLightsCollectionChanged(object sender, NotifyCollectionChangedEventArgs args)
        {
            if (args.Action == NotifyCollectionChangedAction.Add)
            {
                for(int i = 0; i < args.NewItems.Count; ++i)
                {
                    LightModel light = (args.NewItems[i] as LightModel);
                    light.PropertyChanged += OnLightChanged;
                    // Add new light source
                    if (light.GetType() == typeof(PointLightModel))
                    {
                        light.Handle = Core.world_add_light(light.Name, Core.LightType.POINT);
                        LoadSceneLightData(light.Handle, light as PointLightModel);
                    }
                    else if (light.GetType() == typeof(SpotLightModel))
                    {
                        light.Handle = Core.world_add_light(light.Name, Core.LightType.SPOT);
                        LoadSceneLightData(light.Handle, light as SpotLightModel);
                    }
                    else if (light.GetType() == typeof(DirectionalLightModel))
                    {
                        light.Handle = Core.world_add_light(light.Name, Core.LightType.DIRECTIONAL);
                        loadSceneLightData(light.Handle, light as DirectionalLightModel);
                    }
                    else if (light.GetType() == typeof(EnvmapLightModel))
                    {
                        light.Handle = Core.world_add_light(light.Name, Core.LightType.ENVMAP);
                    }
                }
            } else if(args.Action == NotifyCollectionChangedAction.Remove)
            {
                bool needsRebuild = false;
                IntPtr currScenario = Core.world_get_current_scenario();
                for (int i = 0; i < args.OldItems.Count; ++i)
                {
                    LightModel light = (args.OldItems[i] as LightModel);
                    if (light.IsSelected)
                        needsRebuild = true;
                    light.PropertyChanged -= OnLightChanged;
                    // remove light source
                    if (!Core.world_remove_light(light.Handle))
                        throw new Exception(Core.core_get_dll_error());
                    // Also remove it from the scenario, if it was selected, and update the scene
                    if (light.IsSelected)
                    {
                        if (currScenario == IntPtr.Zero)
                            throw new Exception(Core.core_get_dll_error());
                        if (!Core.scenario_remove_light(currScenario, light.Name))
                            throw new Exception(Core.core_get_dll_error());
                        needsRebuild = true;
                    }
                }

                if (needsRebuild)
                {
                    bool wasRendering = m_models.Renderer.IsRendering;
                    m_models.Renderer.IsRendering = false;
                    if (m_reset.CanExecute(null))
                        m_reset.Execute(null);
                    if (Core.world_reload_current_scenario() == IntPtr.Zero)
                        throw new Exception(Core.core_get_dll_error());
                    m_models.Renderer.IsRendering = wasRendering;
                }
            }
        }

        private void OnMaterialsCollectionChanged(object sender, NotifyCollectionChangedEventArgs args)
        {
            // TODO
        }

        private void OnCamerasCollectionChanged(object sender, NotifyCollectionChangedEventArgs args)
        {
            if (args.Action == NotifyCollectionChangedAction.Add)
            {
                for (int i = 0; i < args.NewItems.Count; ++i)
                {
                    CameraModel camera = (args.NewItems[i] as CameraModel);
                    camera.PropertyChanged += OnCameraChanged;
                    if (camera.GetType() == typeof(PinholeCameraModel))
                    {
                        camera.Handle = Core.world_add_pinhole_camera(camera.Name, new Core.Vec3(camera.Position),
                            new Core.Vec3(camera.ViewDirection), new Core.Vec3(camera.Up), camera.Near,
                            camera.Far, (camera as PinholeCameraModel).Fov);
                    }
                    else if (camera.GetType() == typeof(FocusCameraModel))
                    {
                        float lensRadius = (camera as FocusCameraModel).FocalLength / (2f * (camera as FocusCameraModel).Aperture);
                        camera.Handle = Core.world_add_focus_camera(camera.Name, new Core.Vec3(camera.Position),
                            new Core.Vec3(camera.ViewDirection), new Core.Vec3(camera.Up), camera.Near,
                            camera.Far, (camera as FocusCameraModel).FocalLength,
                            (camera as FocusCameraModel).FocusDistance, lensRadius,
                            (camera as FocusCameraModel).SensorHeight);
                    }
                    if (camera.Handle == IntPtr.Zero)
                        throw new Exception(Core.core_get_dll_error());
                }
            } else if(args.Action == NotifyCollectionChangedAction.Remove)
            {
                if(m_models.Cameras.Models.Count == 0)
                {
                    // TODO: Avoid removing the last camera
                }
                for (int i = 0; i < args.OldItems.Count; ++i)
                {
                    CameraModel camera = (args.OldItems[i] as CameraModel);
                    camera.PropertyChanged -= OnCameraChanged;

                    if (camera.IsSelected)
                    {
                        // Change the scenario's camera to the next lower
                        if (args.OldStartingIndex == 0)
                            m_models.Cameras.Models[args.OldStartingIndex].IsSelected = true;
                        else
                            m_models.Cameras.Models[args.OldStartingIndex - 1].IsSelected = true;
                        bool wasRendering = m_models.Renderer.IsRendering;
                        m_models.Renderer.IsRendering = false;
                        if (m_reset.CanExecute(null))
                            m_reset.Execute(null);
                        if (Core.world_reload_current_scenario() == IntPtr.Zero)
                            throw new Exception(Core.core_get_dll_error());
                        m_models.Renderer.IsRendering = wasRendering;
                    }
                }
            }
        }

        private void OnLightChanged(object sender, PropertyChangedEventArgs args)
        {
            LightModel light = (sender as LightModel);
            bool needReload = light.IsSelected || args.PropertyName == nameof(LightModel.IsSelected);
            // TODO: set name!
            if (light.GetType() == typeof(PointLightModel))
            {
                if (args.PropertyName == nameof(PointLightModel.Position) && !Core.world_set_point_light_position(light.Handle, new Core.Vec3((light as PointLightModel).Position)))
                    throw new Exception(Core.core_get_dll_error());
                if (args.PropertyName == nameof(PointLightModel.Intensity) && !Core.world_set_point_light_intensity(light.Handle, new Core.Vec3((light as PointLightModel).Intensity)))
                    throw new Exception(Core.core_get_dll_error());
            }
            else if (light.GetType() == typeof(SpotLightModel))
            {
                if (args.PropertyName == nameof(SpotLightModel.Position) && !Core.world_set_spot_light_position(light.Handle, new Core.Vec3((light as SpotLightModel).Position)))
                    throw new Exception(Core.core_get_dll_error());
                if (args.PropertyName == nameof(SpotLightModel.Direction) && !Core.world_set_spot_light_direction(light.Handle, new Core.Vec3((light as SpotLightModel).Direction)))
                    throw new Exception(Core.core_get_dll_error());
                if (args.PropertyName == nameof(SpotLightModel.Intensity) && !Core.world_set_spot_light_intensity(light.Handle, new Core.Vec3((light as SpotLightModel).Intensity)))
                    throw new Exception(Core.core_get_dll_error());
                if (args.PropertyName == nameof(SpotLightModel.Width) && !Core.world_set_spot_light_angle(light.Handle, (light as SpotLightModel).Width))
                    throw new Exception(Core.core_get_dll_error());
                if (args.PropertyName == nameof(SpotLightModel.FalloffStart) && !Core.world_set_spot_light_falloff(light.Handle, (light as SpotLightModel).FalloffStart))
                    throw new Exception(Core.core_get_dll_error());
            }
            else if (light.GetType() == typeof(DirectionalLightModel))
            {
                if (args.PropertyName == nameof(DirectionalLightModel.Direction) && !Core.world_set_dir_light_direction(light.Handle, new Core.Vec3((light as DirectionalLightModel).Direction)))
                    throw new Exception(Core.core_get_dll_error());
                if (args.PropertyName == nameof(DirectionalLightModel.Irradiance) && !Core.world_set_dir_light_irradiance(light.Handle, new Core.Vec3((light as DirectionalLightModel).Irradiance)))
                    throw new Exception(Core.core_get_dll_error());
            }
            else if (light.GetType() == typeof(EnvmapLightModel))
            {
                if(args.PropertyName == nameof(EnvmapLightModel.Map))
                {
                    string absoluteTexturePath = Path.Combine(m_models.Scene.Directory, (light as EnvmapLightModel).Map);
                    IntPtr envMapHandle = Core.world_add_texture(absoluteTexturePath, Core.TextureSampling.LINEAR);
                    if (envMapHandle == IntPtr.Zero)
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.world_set_env_light_map(light.Handle, envMapHandle))
                        throw new Exception(Core.core_get_dll_error());
                }
            }

            IntPtr currScenario = Core.world_get_current_scenario();
            if (currScenario == IntPtr.Zero)
                throw new Exception(Core.core_get_dll_error());

            if (args.PropertyName == nameof(LightModel.IsSelected))
            {
                if (light.IsSelected)
                {
                    if (!Core.scenario_add_light(currScenario, light.Name))
                        throw new Exception(Core.core_get_dll_error());
                }
                else
                {
                    if (!Core.scenario_remove_light(currScenario, light.Name))
                        throw new Exception(Core.core_get_dll_error());
                }
            }

            if (needReload)
            {
                bool wasRendering = m_models.Renderer.IsRendering;
                m_models.Renderer.IsRendering = false;
                if (m_reset.CanExecute(null))
                    m_reset.Execute(null);
                if (Core.world_reload_current_scenario() == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());
                m_models.Renderer.IsRendering = wasRendering;
            }
        }

        private void OnMaterialChanged(object sender, PropertyChangedEventArgs args)
        {
            MaterialModel material = (sender as MaterialModel);
            // TODO

            bool needReload = false;

            if(needReload)
            {
                bool wasRendering = m_models.Renderer.IsRendering;
                m_models.Renderer.IsRendering = false;
                if (m_reset.CanExecute(null))
                    m_reset.Execute(null);
                if (Core.world_reload_current_scenario() == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());
                m_models.Renderer.IsRendering = wasRendering;
            }
        }

        private void OnCameraChanged(object sender, PropertyChangedEventArgs args)
        {
            // TODO: we can find out what was changed easily
            CameraModel camera = (sender as CameraModel);
            // We only reload the renderer if the changed camera is selected or becomes selected;
            // since the handler will be called twice (once for the selection and once for the deselection),
            // we shouldn't reset both times
            bool needReload = camera.IsSelected;

            // TODO: set name!
            // TODO: how to move/rotate camera?
            if (args.PropertyName == nameof(CameraModel.Position) && !Core.world_set_camera_position(camera.Handle, new Core.Vec3(camera.Position)))
                throw new Exception(Core.core_get_dll_error());
            if (args.PropertyName == nameof(CameraModel.Near) && !Core.world_set_camera_near(camera.Handle, camera.Near))
                throw new Exception(Core.core_get_dll_error());
            if (args.PropertyName == nameof(CameraModel.Far) && !Core.world_set_camera_far(camera.Handle, camera.Far))
                throw new Exception(Core.core_get_dll_error());
            if (camera.GetType() == typeof(PinholeCameraModel))
            {
                if (args.PropertyName == nameof(PinholeCameraModel.Fov) && !Core.world_set_pinhole_camera_fov(camera.Handle, (camera as PinholeCameraModel).Fov))
                    throw new Exception(Core.core_get_dll_error());
            } else if(camera.GetType() == typeof(FocusCameraModel))
            {
                if (args.PropertyName == nameof(FocusCameraModel.FocalLength) && !Core.world_set_focus_camera_focal_length(camera.Handle, (camera as FocusCameraModel).FocalLength))
                    throw new Exception(Core.core_get_dll_error());
                if (args.PropertyName == nameof(FocusCameraModel.FocusDistance) && !Core.world_set_focus_camera_focus_distance(camera.Handle, (camera as FocusCameraModel).FocusDistance))
                    throw new Exception(Core.core_get_dll_error());
                if (args.PropertyName == nameof(FocusCameraModel.SensorHeight) && !Core.world_set_focus_camera_sensor_height(camera.Handle, (camera as FocusCameraModel).SensorHeight))
                    throw new Exception(Core.core_get_dll_error());
                if (args.PropertyName == nameof(FocusCameraModel.Aperture) && !Core.world_set_focus_camera_aperture(camera.Handle, (camera as FocusCameraModel).Aperture))
                    throw new Exception(Core.core_get_dll_error());
            }

            IntPtr currScenario = Core.world_get_current_scenario();
            if(currScenario == IntPtr.Zero)
                throw new Exception(Core.core_get_dll_error());
            if (camera.IsSelected)
                if (!Core.scenario_set_camera(currScenario, camera.Handle))
                    throw new Exception(Core.core_get_dll_error());

            if (needReload)
            {
                bool wasRendering = m_models.Renderer.IsRendering;
                m_models.Renderer.IsRendering = false;
                if (m_reset.CanExecute(null))
                    m_reset.Execute(null);
                if (Core.world_reload_current_scenario() == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());
                m_models.Renderer.IsRendering = wasRendering;
            }
        }

        private void OnSceneChanged(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(SceneModel.FullPath):
                    {
                        if(m_models.Scene.FullPath != null)
                        {
                            // Temporarily disable handlers
                            m_models.Lights.Models.CollectionChanged -= OnLightsCollectionChanged;
                            m_models.Materials.Models.CollectionChanged -= OnMaterialsCollectionChanged;
                            m_models.Cameras.Models.CollectionChanged -= OnCamerasCollectionChanged;
                            LoadSceneLights();
                            LoadSceneMaterials();
                            LoadSceneCameras();
                            m_models.Lights.Models.CollectionChanged += OnLightsCollectionChanged;
                            m_models.Materials.Models.CollectionChanged += OnMaterialsCollectionChanged;
                            m_models.Cameras.Models.CollectionChanged += OnCamerasCollectionChanged;
                        }
                    }   break;
                case nameof(SceneModel.CurrentScenario):
                    LoadScenarioViews();
                    break;
            }
        }

        private void ScenariosOnCollectionChanged(object sender, NotifyCollectionChangedEventArgs e)
        {
            // TODO make more efficient for single value insertions
            MakeScenerioViews();
            /*IntPtr currentScenario = Core.world_get_current_scenario();
            if (currentScenario == IntPtr.Zero)
                throw new Exception(Core.core_get_dll_error());
            string currentScenarioName = Core.world_get_scenario_name(currentScenario);
            m_scenarioBox.SelectedItem = null;

            // Select the loaded scenario (unsubscribe the event since that scenario is already loaded)
            m_scenarioBox.SelectionChanged -= ScenarioChanged;
            foreach (var item in m_scenarioBox.Items)
            {
                if ((item as string) == currentScenarioName)
                {
                    m_scenarioBox.SelectedItem = item;
                    break;
                }
            }
            LoadScenarioLights(currentScenario);
            loadScenarioMaterials(currentScenario);
            LoadScenarioCamera(currentScenario);
            m_scenarioBox.SelectionChanged += ScenarioChanged;*/
        }

        /// <summary>
        /// clears Scenarios collection and fills it with the new values.
        /// Sets SelectedScenario as well.
        /// </summary>
        private void MakeScenerioViews()
        {
            Scenarios.Clear();

            if (m_models.Scene == null)
            {
                SelectedScenario = null;
                return;
            }

            // code assumes that an active scenario exists
            Debug.Assert(m_models.Scene.CurrentScenario != null);

            foreach (var scenario in m_models.Scene.Scenarios)
            {
                // add scenario view
                var view = new ComboBoxItem<ScenarioModel>(scenario.Name, scenario);
                Scenarios.Add(view);
                // set selected item
                if (ReferenceEquals(scenario, m_models.Scene.CurrentScenario))
                    SelectedScenario = view;
            }
            LoadScenarioViews();
        }

        private void LoadScenarioViews()
        {
            // First fetch the scenario-specific information
            LoadScenarioLights(m_models.Scene.CurrentScenario.Handle);
            loadScenarioMaterials(m_models.Scene.CurrentScenario.Handle);
            LoadScenarioCamera(m_models.Scene.CurrentScenario.Handle);

            // Change display and renderer resolution
            m_models.Viewport.RenderWidth = (int)m_models.Scene.CurrentScenario.Resolution.X;
            m_models.Viewport.RenderHeight = (int)m_models.Scene.CurrentScenario.Resolution.Y;
            //m_scenarioLoadDialog.Close();
        }

        private void LoadScenarioLights(IntPtr scenarioHdl)
        {
            foreach (var light in m_models.Lights.Models)
            {
                light.PropertyChanged -= OnLightChanged;
                light.IsSelected = false;
                light.PropertyChanged += OnLightChanged;
            }
            { // Point lights
                int count = Core.scenario_get_point_light_count(scenarioHdl);
                for (int i = 0; i < count; ++i)
                {
                    string name = Core.scenario_get_point_light_name(scenarioHdl, (ulong)i);
                    if (name == null || name.Length == 0)
                        throw new Exception(Core.core_get_dll_error());
                    foreach (var light in m_models.Lights.Models)
                    {
                        if (light.Name == name)
                        {
                            light.PropertyChanged -= OnLightChanged;
                            light.IsSelected = true;
                            light.PropertyChanged += OnLightChanged;
                            break;
                        }
                    }
                }
            }
            { // Spot lights
                int count = Core.scenario_get_spot_light_count(scenarioHdl);
                for (int i = 0; i < count; ++i)
                {
                    string name = Core.scenario_get_spot_light_name(scenarioHdl, (ulong)i);
                    if (name == null || name.Length == 0)
                        throw new Exception(Core.core_get_dll_error());
                    foreach (var light in m_models.Lights.Models)
                    {
                        if (light.Name == name)
                        {
                            light.PropertyChanged -= OnLightChanged;
                            light.IsSelected = true;
                            light.PropertyChanged += OnLightChanged;
                            break;
                        }
                    }
                }
            }
            { // Directional lights
                int count = Core.scenario_get_dir_light_count(scenarioHdl);
                for (int i = 0; i < count; ++i)
                {
                    string name = Core.scenario_get_dir_light_name(scenarioHdl, (ulong)i);
                    if (name == null || name.Length == 0)
                        throw new Exception(Core.core_get_dll_error());
                    foreach (var light in m_models.Lights.Models)
                    {
                        if (light.Name == name)
                        {
                            light.PropertyChanged -= OnLightChanged;
                            light.IsSelected = true;
                            light.PropertyChanged += OnLightChanged;
                            break;
                        }
                    }
                }
            }
            { // Envmap light
                if(Core.scenario_has_envmap_light(scenarioHdl))
                {
                    string name = Core.scenario_get_envmap_light_name(scenarioHdl);
                    if (name == null || name.Length == 0)
                        throw new Exception(Core.core_get_dll_error());
                    foreach (var light in m_models.Lights.Models)
                    {
                        if (light.Name == name)
                        {
                            light.PropertyChanged -= OnLightChanged;
                            light.IsSelected = true;
                            light.PropertyChanged += OnLightChanged;
                            break;
                        }
                    }
                }
            }
        }

        private void loadScenarioMaterials(IntPtr scenarioHdl)
        {
            // TODO
        }

        private void LoadScenarioCamera(IntPtr scenarioHdl)
        {
            IntPtr cam = Core.scenario_get_camera(scenarioHdl);
            if (cam == IntPtr.Zero)
                throw new Exception(Core.core_get_dll_error());
            string name = Core.world_get_camera_name(cam);
            if (name == null || name.Length == 0)
                throw new Exception(Core.core_get_dll_error());

            foreach (var camera in m_models.Cameras.Models)
            {
                camera.PropertyChanged -= OnCameraChanged;
                if (name == camera.Name)
                {
                    camera.IsSelected = true;
                }
                else
                {
                    camera.IsSelected = false;
                }
                camera.PropertyChanged += OnCameraChanged;
            }
        }

        private static void LoadSceneLightData(IntPtr lightHdl, PointLightModel lightModel)
        {
            Core.Vec3 pos = new Core.Vec3();
            Core.Vec3 intensity = new Core.Vec3();
            if (!Core.world_get_point_light_position(lightHdl, ref pos))
                throw new Exception(Core.core_get_dll_error());
            if (!Core.world_get_point_light_intensity(lightHdl, ref intensity))
                throw new Exception(Core.core_get_dll_error());

            lightModel.Position = new Vec3<float>(pos.x, pos.y, pos.z);
            lightModel.Intensity = new Vec3<float>(intensity.x, intensity.y, intensity.z);
        }

        private void LoadSceneLightData(IntPtr lightHdl, SpotLightModel lightModel)
        {
            Core.Vec3 pos = new Core.Vec3();
            Core.Vec3 dir = new Core.Vec3();
            Core.Vec3 intensity = new Core.Vec3();
            float angle = 0f;
            float falloff = 0f;
            if (!Core.world_get_spot_light_position(lightHdl, ref pos))
                throw new Exception(Core.core_get_dll_error());
            if (!Core.world_get_spot_light_direction(lightHdl, ref dir))
                throw new Exception(Core.core_get_dll_error());
            if (!Core.world_get_spot_light_intensity(lightHdl, ref intensity))
                throw new Exception(Core.core_get_dll_error());
            if (!Core.world_get_spot_light_angle(lightHdl, ref angle))
                throw new Exception(Core.core_get_dll_error());
            if (!Core.world_get_spot_light_falloff(lightHdl, ref falloff))
                throw new Exception(Core.core_get_dll_error());

            lightModel.Position = new Vec3<float>(pos.x, pos.y, pos.z);
            lightModel.Direction = new Vec3<float>(dir.x, dir.y, dir.z);
            lightModel.Intensity = new Vec3<float>(intensity.x, intensity.y, intensity.z);
            lightModel.Width = angle;
            lightModel.FalloffStart = falloff;
        }

        private void loadSceneLightData(IntPtr lightHdl, DirectionalLightModel lightModel)
        {
            Core.Vec3 dir = new Core.Vec3();
            Core.Vec3 irradiance = new Core.Vec3();
            if (!Core.world_get_dir_light_direction(lightHdl, ref dir))
                throw new Exception(Core.core_get_dll_error());
            if (!Core.world_get_dir_light_irradiance(lightHdl, ref irradiance))
                throw new Exception(Core.core_get_dll_error());

            lightModel.Direction = new Vec3<float>(dir.x, dir.y, dir.z);
            lightModel.Irradiance = new Vec3<float>(irradiance.x, irradiance.y, irradiance.z);
        }

        private void LoadSceneLights()
        {
            m_models.Lights.Models.Clear();

            if(m_models.Scene == null) return;
            // Point lights
            ulong pointLightCount = Core.world_get_point_light_count();
            for(ulong i = 0u; i < pointLightCount; ++i)
            {
                IntPtr hdl = Core.world_get_light_handle(i, Core.LightType.POINT);
                string name = Core.world_get_light_name(hdl);
                if (name == null || name.Length == 0)
                    throw new Exception(Core.core_get_dll_error());

                m_models.Lights.Models.Add(new PointLightModel()
                {
                    Name = name,
                    Handle = hdl,
                });
                LoadSceneLightData(hdl, m_models.Lights.Models.Last() as PointLightModel);
                m_models.Lights.Models.Last().PropertyChanged += OnLightChanged;
            }

            // Spot lights
            ulong spotLightCount = Core.world_get_spot_light_count();
            for (ulong i = 0u; i < spotLightCount; ++i)
            {
                IntPtr hdl = Core.world_get_light_handle(i, Core.LightType.SPOT);
                string name = Core.world_get_light_name(hdl);
                if (name == null || name.Length == 0)
                    throw new Exception(Core.core_get_dll_error());

                m_models.Lights.Models.Add(new SpotLightModel()
                {
                    Name = name,
                    Handle = hdl,
                });
                LoadSceneLightData(hdl, m_models.Lights.Models.Last() as SpotLightModel);
                m_models.Lights.Models.Last().PropertyChanged += OnLightChanged;
            }

            // Directional lights
            ulong dirLightCount = Core.world_get_dir_light_count();
            for (ulong i = 0u; i < dirLightCount; ++i)
            {
                IntPtr hdl = Core.world_get_light_handle(i, Core.LightType.DIRECTIONAL);
                string name = Core.world_get_light_name(hdl);
                if (name == null || name.Length == 0)
                    throw new Exception(Core.core_get_dll_error());

                m_models.Lights.Models.Add(new DirectionalLightModel()
                {
                    Name = name,
                    Handle = hdl,
                });
                loadSceneLightData(hdl, m_models.Lights.Models.Last() as DirectionalLightModel);
                m_models.Lights.Models.Last().PropertyChanged += OnLightChanged;
            }

            // Envmap lights
            ulong envLightCount = Core.world_get_env_light_count();
            for (ulong i = 0u; i < envLightCount; ++i)
            {
                IntPtr hdl = Core.world_get_light_handle(i, Core.LightType.ENVMAP);
                string name = Core.world_get_light_name(hdl);
                if (name == null || name.Length == 0)
                    throw new Exception(Core.core_get_dll_error());
                string absoluteTexPath = Core.world_get_env_light_map(hdl);
                if (absoluteTexPath == null || absoluteTexPath.Length == 0)
                    throw new Exception(Core.core_get_dll_error());
                Uri scenePath = new Uri(m_models.Scene.FullPath);
                string map = scenePath.MakeRelativeUri(new Uri(absoluteTexPath)).OriginalString;
                m_models.Lights.Models.Add(new EnvmapLightModel()
                {
                    Name = name,
                    Handle = hdl,
                    Map = map
                });
            }
        }

        private void LoadSceneMaterials()
        {
            // TODO
            m_models.Materials.Models.Clear();
        }

        private void LoadSceneCameras()
        {
            m_models.Cameras.Models.Clear();
            ulong count = Core.world_get_camera_count();
            for(ulong i = 0u; i < count; ++i)
            {
                IntPtr cam = Core.world_get_camera_by_index(i);
                if (cam == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());
                string name = Core.world_get_camera_name(cam);
                if (name == null || name.Length == 0)
                    throw new Exception(Core.core_get_dll_error());
                Core.CameraType type = Core.world_get_camera_type(cam);
                CameraModel model = null;
                switch (type)
                {
                    case Core.CameraType.PINHOLE:
                        {
                            float fov = 0f;
                            if (!Core.world_get_pinhole_camera_fov(cam, ref fov))
                                throw new Exception(Core.core_get_dll_error());

                            model = new PinholeCameraModel()
                            {
                                Fov = fov
                            };
                        }
                        break;
                    case Core.CameraType.FOCUS:
                        {
                            float focalLength = 0f;
                            float focusDistance = 0f;
                            float sensorHeight = 0f;
                            float aperture = 0f;
                            if (!Core.world_get_focus_camera_focal_length(cam, ref focalLength))
                                throw new Exception(Core.core_get_dll_error());
                            if (!Core.world_get_focus_camera_focus_distance(cam, ref focusDistance))
                                throw new Exception(Core.core_get_dll_error());
                            if (!Core.world_get_focus_camera_sensor_height(cam, ref sensorHeight))
                                throw new Exception(Core.core_get_dll_error());
                            if (!Core.world_get_focus_camera_aperture(cam, ref aperture))
                                throw new Exception(Core.core_get_dll_error());

                            model = new FocusCameraModel()
                            {
                                FocalLength = focalLength,
                                FocusDistance = focusDistance,
                                SensorHeight = sensorHeight,
                                Aperture = aperture
                            };
                        }
                        break;
                }
                if (model != null)
                {
                    Core.Vec3 pos = new Core.Vec3();
                    Core.Vec3 viewDir = new Core.Vec3();
                    Core.Vec3 up = new Core.Vec3();
                    float near = 0f;
                    float far = 0f;
                    if (!Core.world_get_camera_position(cam, ref pos))
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.world_get_camera_direction(cam, ref viewDir))
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.world_get_camera_up(cam, ref up))
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.world_get_camera_near(cam, ref near))
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.world_get_camera_far(cam, ref far))
                        throw new Exception(Core.core_get_dll_error());
                    model.Position = new Vec3<float>(pos.x, pos.y, pos.z);
                    model.ViewDirection = new Vec3<float>(viewDir.x, viewDir.y, viewDir.z);
                    model.Up = new Vec3<float>(up.x, up.y, up.z);
                    model.Near = near;
                    model.Far = far;
                    model.Name = name;
                    model.Handle = cam;
                    m_models.Cameras.Models.Add(model);
                }
            }
        }

        private void OnRendererChange(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(RendererModel.IsRendering):
                    OnPropertyChanged(nameof(CanLoadLastScenes));
                    break;
            }
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
