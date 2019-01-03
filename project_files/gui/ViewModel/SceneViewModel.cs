using System;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using gui.Annotations;
using gui.Dll;
using gui.Model;
using gui.Model.Camera;
using gui.Model.Light;
using gui.Model.Material;
using gui.Properties;
using gui.Utility;
using gui.View;

namespace gui.ViewModel
{
    public class SceneViewModel : INotifyPropertyChanged
    {
        public class SceneMenuItem : ICommand
        {
            private Models m_models;

            public string Filename { get; set; }
            public string Path { get; set; }
            public ICommand Command { get; }

            public SceneMenuItem(Models models)
            {
                m_models = models;
                Command = this;
            }

            SceneMenuItem(ICommand command)
            {
                Command = command;
            }

            public bool CanExecute(object parameter)
            {
                return !m_models.Renderer.IsRendering;
            }

            public void Execute(object parameter)
            {
                m_models.Scene.LoadScene(Path);
            }

            public event EventHandler CanExecuteChanged
            {
                add { }
                remove { }
            }
        }

        private Models m_models;
        private ListBox m_scenarioBox;
        private ScenarioLoadStatus m_scenarioLoadDialog;
        public ObservableCollection<SceneMenuItem> LastScenes { get; }
        public ObservableCollection<string> Scenarios { get => m_models.Scene.Scenarios; }
        public bool CanLoadLastScenes { get => LastScenes.Count > 0 && !m_models.Renderer.IsRendering; }

        public SceneViewModel(MainWindow window, Models models)
        {
            m_models = models;
            m_scenarioBox = (ListBox)window.FindName("ScenarioSelectionBox");
            m_scenarioBox.SelectionChanged += scenarioChanged;
            LastScenes = new ObservableCollection<SceneMenuItem>();
            foreach (string path in m_models.Scene.LastScenes)
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

            m_models.Scene.PropertyChanged += changeScene;
            m_models.Renderer.PropertyChanged += renderStatusChanged;
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
                        light.Handle = Core.world_add_point_light(light.Name, new Core.Vec3((light as PointLightModel).Position),
                            new Core.Vec3((light as PointLightModel).Intensity));
                    }
                    else if (light.GetType() == typeof(SpotLightModel))
                    {
                        light.Handle = Core.world_add_spot_light(light.Name, new Core.Vec3((light as SpotLightModel).Position),
                            new Core.Vec3((light as SpotLightModel).Direction), new Core.Vec3((light as SpotLightModel).Intensity),
                            (light as SpotLightModel).Width, (light as SpotLightModel).FalloffStart);
                    }
                    else if (light.GetType() == typeof(DirectionalLightModel))
                    {
                        light.Handle = Core.world_add_directional_light(light.Name, new Core.Vec3((light as DirectionalLightModel).Direction),
                            new Core.Vec3((light as DirectionalLightModel).Radiance));
                    }
                    else if (light.GetType() == typeof(EnvmapLightModel))
                    {
                        IntPtr envMapHandle = Core.world_add_texture((light as EnvmapLightModel).Map, Core.TextureSampling.LINEAR);
                        if (envMapHandle == IntPtr.Zero)
                            throw new Exception(Core.core_get_dll_error());
                        light.Handle = Core.world_add_envmap_light(light.Name, envMapHandle);
                    }
                    if (light.Handle == IntPtr.Zero)
                        throw new Exception(Core.core_get_dll_error());
                }
            } else if(args.Action == NotifyCollectionChangedAction.Remove)
            {
                bool wasRendering = m_models.Renderer.IsRendering;
                bool needsRebuild = false;
                IntPtr currScenario = Core.world_get_current_scenario();
                for (int i = 0; i < args.NewItems.Count; ++i)
                {
                    LightModel light = (args.OldItems[i] as LightModel);
                    if(light.IsSelected)
                    {
                        m_models.Renderer.IsRendering = false;
                        needsRebuild = true;
                    }
                    light.PropertyChanged -= OnLightChanged;
                    // remove light source
                    if (light.GetType() == typeof(PointLightModel))
                    {
                        if (!Core.world_remove_point_light((light as PointLightModel).Handle))
                            throw new Exception(Core.core_get_dll_error());
                    }
                    else if (light.GetType() == typeof(SpotLightModel))
                    {
                        if (!Core.world_remove_spot_light((light as SpotLightModel).Handle))
                            throw new Exception(Core.core_get_dll_error());
                    }
                    else if (light.GetType() == typeof(DirectionalLightModel))
                    {
                        if (!Core.world_remove_dir_light((light as DirectionalLightModel).Handle))
                            throw new Exception(Core.core_get_dll_error());
                    }
                    else if (light.GetType() == typeof(EnvmapLightModel))
                    {
                        if (!Core.world_remove_envmap_light((light as EnvmapLightModel).Handle))
                            throw new Exception(Core.core_get_dll_error());
                    }
                    // Also remove it from the scenario, if it was selected, and update the scene
                    if ((light as LightModel).IsSelected)
                    {
                        if (currScenario == IntPtr.Zero)
                            throw new Exception(Core.core_get_dll_error());
                        if (!Core.scenario_remove_light_by_named(currScenario, (light as LightModel).Name))
                            throw new Exception(Core.core_get_dll_error());
                        if (light.GetType() == typeof(EnvmapLightModel))
                            Core.scene_mark_envmap_dirty();
                        else
                            Core.scene_mark_lighttree_dirty();
                        needsRebuild = true;
                    }
                }

                if(needsRebuild)
                {
                    if (Core.world_reload_current_scenario() == IntPtr.Zero)
                        throw new Exception(Core.core_get_dll_error());
                    m_models.Renderer.reset();
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
                        bool wasRendering = m_models.Renderer.IsRendering;
                        m_models.Renderer.IsRendering = false;
                        // Change the scenario's camera to the next lower
                        if (args.OldStartingIndex == 0)
                            m_models.Cameras.Models[args.OldStartingIndex].IsSelected = true;
                        else
                            m_models.Cameras.Models[args.OldStartingIndex - 1].IsSelected = true;
                        if (Core.world_reload_current_scenario() == IntPtr.Zero)
                            throw new Exception(Core.core_get_dll_error());
                        m_models.Renderer.reset();
                        m_models.Renderer.IsRendering = wasRendering;
                    }
                }
            }
        }

        private void OnLightChanged(object sender, PropertyChangedEventArgs args)
        {
            // Lights need to reset and rebuild the scene
            bool wasRendering = m_models.Renderer.IsRendering;
            LightModel light = (sender as LightModel);
            bool needReload = light.IsSelected || args.PropertyName == nameof(LightModel.IsSelected);
            if (needReload)
            {
                m_models.Renderer.IsRendering = false;
                m_models.Renderer.reset();
            }
            // TODO: set name!
            if (light.GetType() == typeof(PointLightModel))
            {
                if (args.PropertyName == nameof(PointLightModel.Position) && !Core.world_set_point_light_position(light.Handle, new Core.Vec3((light as PointLightModel).Position)))
                    throw new Exception(Core.core_get_dll_error());
                if (args.PropertyName == nameof(PointLightModel.Intensity) && !Core.world_set_point_light_intensity(light.Handle, new Core.Vec3((light as PointLightModel).Intensity)))
                    throw new Exception(Core.core_get_dll_error());
                Core.scene_mark_lighttree_dirty();
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
                Core.scene_mark_lighttree_dirty();
            }
            else if (light.GetType() == typeof(DirectionalLightModel))
            {
                if (args.PropertyName == nameof(DirectionalLightModel.Direction) && !Core.world_set_dir_light_direction(light.Handle, new Core.Vec3((light as DirectionalLightModel).Direction)))
                    throw new Exception(Core.core_get_dll_error());
                if (args.PropertyName == nameof(DirectionalLightModel.Radiance) && !Core.world_set_dir_light_radiance(light.Handle, new Core.Vec3((light as DirectionalLightModel).Radiance)))
                    throw new Exception(Core.core_get_dll_error());
                Core.scene_mark_lighttree_dirty();
            }
            else if (light.GetType() == typeof(EnvmapLightModel))
            {
                if(args.PropertyName == nameof(EnvmapLightModel.Map))
                {
                    IntPtr envMapHandle = Core.world_add_texture((light as EnvmapLightModel).Map, Core.TextureSampling.LINEAR);
                    if (envMapHandle == IntPtr.Zero)
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.world_set_env_light_map(light.Handle, envMapHandle))
                        throw new Exception(Core.core_get_dll_error());
                }
                Core.scene_mark_envmap_dirty();
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
                    if (!Core.scenario_remove_light_by_named(currScenario, light.Name))
                        throw new Exception(Core.core_get_dll_error());
                }
            }

            if (needReload)
            {
                if (Core.world_reload_current_scenario() == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());
                m_models.Renderer.IsRendering = wasRendering;
            }
        }

        private void OnMaterialChanged(object sender, PropertyChangedEventArgs args)
        {
            MaterialModel material = (sender as MaterialModel);
            // TODO

            IntPtr currScenario = Core.world_get_current_scenario();
            // Materials need to reset and rebuild the scene
            if (currScenario == IntPtr.Zero)
                throw new Exception(Core.core_get_dll_error());
            if (Core.world_load_scenario(currScenario) == IntPtr.Zero)
                throw new Exception(Core.core_get_dll_error());
            m_models.Renderer.reset();
        }


        private void OnCameraChanged(object sender, PropertyChangedEventArgs args)
        {
            // TODO: we can find out what was changed easily
            CameraModel camera = (sender as CameraModel);
            bool wasRendering = m_models.Renderer.IsRendering;
            // We only reload the renderer if the changed camera is selected or becomes selected;
            // since the handler will be called twice (once for the selection and once for the deselection),
            // we shouldn't reset both times
            bool needReload = camera.IsSelected;
            if (needReload)
            {
                m_models.Renderer.IsRendering = false;
                // Cameras only need to reset
                m_models.Renderer.reset();
            }

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
                if (Core.world_reload_current_scenario() == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());
                m_models.Renderer.IsRendering = wasRendering;
            }
        }

        private void changeScene(object sender, PropertyChangedEventArgs args)
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
                            loadSceneLights();
                            loadSceneMaterials();
                            loadSceneCameras();
                            m_models.Lights.Models.CollectionChanged += OnLightsCollectionChanged;
                            m_models.Materials.Models.CollectionChanged += OnMaterialsCollectionChanged;
                            m_models.Cameras.Models.CollectionChanged += OnCamerasCollectionChanged;
                        }
                    }   break;
                case nameof(SceneModel.LastScenes):
                    {
                        LastScenes.Clear();
                        foreach(string path in m_models.Scene.LastScenes)
                        {
                            LastScenes.Add(new SceneMenuItem(m_models)
                            {
                                Filename = Path.GetFileName(path),
                                Path = path
                            });
                        }
                        OnPropertyChanged(nameof(CanLoadLastScenes));
                        
                    }   break;
                case nameof(SceneModel.Scenarios):
                    {
                        IntPtr currentScenario = Core.world_get_current_scenario();
                        if (currentScenario == IntPtr.Zero)
                            throw new Exception(Core.core_get_dll_error());
                        string currentScenarioName = Core.world_get_scenario_name(currentScenario);
                        m_scenarioBox.SelectedItem = null;

                        // Select the loaded scenario (unsubscribe the event since that scenario is already loaded)
                        m_scenarioBox.SelectionChanged -= scenarioChanged;
                        foreach (var item in m_scenarioBox.Items)
                        {
                            if ((item as string) == currentScenarioName)
                            {
                                m_scenarioBox.SelectedItem = item;
                                break;
                            }
                        }
                        loadScenarioLights(currentScenario);
                        loadScenarioMaterials(currentScenario);
                        loadScenarioCamera(currentScenario);
                        m_scenarioBox.SelectionChanged += scenarioChanged;
                    }   break;

            }
        }

        private void scenarioChanged(object sender, SelectionChangedEventArgs args)
        {
            if(args.AddedItems.Count > 0)
            {
                IntPtr scenario = Core.world_find_scenario(args.AddedItems[0] as string);
                if (scenario == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());
                // The dialog will be closed when all scenario work is done (at the end of loadScenarioViews)
                m_scenarioLoadDialog = new ScenarioLoadStatus(args.AddedItems[0] as string);
                loadScenarioAsync(scenario);
                m_models.Scene.CurrentScenario = args.AddedItems[0] as string;
            }
        }

        private async void loadScenarioAsync(IntPtr scenario)
        {
            await Task.Run(() =>
            {
                if (Core.world_load_scenario(scenario) == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());
                Application.Current.Dispatcher.BeginInvoke(new Action(() => loadScenarioViews(scenario)));
            });
        }

        private void loadScenarioViews(IntPtr scenarioHdl)
        {
            // First fetch the scenario-specific information
            loadScenarioLights(scenarioHdl);
            loadScenarioMaterials(scenarioHdl);
            loadScenarioCamera(scenarioHdl);

            m_scenarioLoadDialog.Close();
        }

        private void loadScenarioLights(IntPtr scenarioHdl)
        {
            foreach (var light in m_models.Lights.Models)
            {
                light.PropertyChanged -= OnLightChanged;
                light.IsSelected = false;
                light.PropertyChanged += OnLightChanged;
            }
            ulong count = Core.scenario_get_light_count(scenarioHdl);
            for(ulong i = 0u; i < count; ++i)
            {
                string name = Core.scenario_get_light_name(scenarioHdl, i);
                if(name == null || name.Length == 0)
                    throw new Exception(Core.core_get_dll_error());
                foreach (var light in m_models.Lights.Models)
                {
                    if(light.Name == name)
                    {
                        light.PropertyChanged -= OnLightChanged;
                        light.IsSelected = true;
                        light.PropertyChanged += OnLightChanged;
                        break;
                    }
                }
            }
        }

        private void loadScenarioMaterials(IntPtr scenarioHdl)
        {
            // TODO
        }

        private void loadScenarioCamera(IntPtr scenarioHdl)
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

        private void loadSceneLights()
        {
            m_models.Lights.Models.Clear();
            // Point lights
            ulong pointLightCount = Core.world_get_point_light_count();
            for(ulong i = 0u; i < pointLightCount; ++i)
            {
                IntPtr hdl = IntPtr.Zero;
                string name = Core.world_get_point_light_by_index(i, ref hdl);
                if (hdl == IntPtr.Zero || name == null || name.Length == 0)
                    throw new Exception(Core.core_get_dll_error());
                Core.Vec3 pos = new Core.Vec3();
                Core.Vec3 intensity = new Core.Vec3();
                if (!Core.world_get_point_light_position(hdl, ref pos))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_get_point_light_intensity(hdl, ref intensity))
                    throw new Exception(Core.core_get_dll_error());

                m_models.Lights.Models.Add(new PointLightModel()
                {
                    Name = name,
                    Handle = hdl,
                    Position = new Vec3<float>(pos.x, pos.y, pos.z),
                    Intensity = new Vec3<float>(intensity.x, intensity.y, intensity.z)
                });
                m_models.Lights.Models.Last().PropertyChanged += OnLightChanged;
            }

            // Spot lights
            ulong spotLightCount = Core.world_get_spot_light_count();
            for (ulong i = 0u; i < spotLightCount; ++i)
            {
                IntPtr hdl = IntPtr.Zero;
                string name = Core.world_get_spot_light_by_index(i, ref hdl);
                if (hdl == IntPtr.Zero || name == null || name.Length == 0)
                    throw new Exception(Core.core_get_dll_error());
                Core.Vec3 pos = new Core.Vec3();
                Core.Vec3 dir = new Core.Vec3();
                Core.Vec3 intensity = new Core.Vec3();
                float angle = 0f;
                float falloff = 0f;
                if (!Core.world_get_spot_light_position(hdl, ref pos))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_get_spot_light_direction(hdl, ref dir))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_get_spot_light_intensity(hdl, ref intensity))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_get_spot_light_angle(hdl, ref angle))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_get_spot_light_falloff(hdl, ref falloff))
                    throw new Exception(Core.core_get_dll_error());

                m_models.Lights.Models.Add(new SpotLightModel()
                {
                    Name = name,
                    Handle = hdl,
                    Position = new Vec3<float>(pos.x, pos.y, pos.z),
                    Direction = new Vec3<float>(dir.x, dir.y, dir.z),
                    Intensity = new Vec3<float>(intensity.x, intensity.y, intensity.z),
                    Width = angle,
                    FalloffStart = falloff
                });
                m_models.Lights.Models.Last().PropertyChanged += OnLightChanged;
            }

            // Directional lights
            ulong dirLightCount = Core.world_get_dir_light_count();
            for (ulong i = 0u; i < dirLightCount; ++i)
            {
                IntPtr hdl = IntPtr.Zero;
                string name = Core.world_get_dir_light_by_index(i, ref hdl);
                if (hdl == IntPtr.Zero || name == null || name.Length == 0)
                    throw new Exception(Core.core_get_dll_error());
                Core.Vec3 dir = new Core.Vec3();
                Core.Vec3 radiance = new Core.Vec3();
                if (!Core.world_get_dir_light_direction(hdl, ref dir))
                    throw new Exception(Core.core_get_dll_error());
                if (!Core.world_get_dir_light_radiance(hdl, ref radiance))
                    throw new Exception(Core.core_get_dll_error());

                m_models.Lights.Models.Add(new DirectionalLightModel()
                {
                    Name = name,
                    Handle = hdl,
                    Direction = new Vec3<float>(dir.x, dir.y, dir.z),
                    Radiance = new Vec3<float>(radiance.x, radiance.y, radiance.z)
                });
                m_models.Lights.Models.Last().PropertyChanged += OnLightChanged;
            }

            // Envmap lights
            ulong envLightCount = Core.world_get_env_light_count();
            for (ulong i = 0u; i < envLightCount; ++i)
            {
                IntPtr hdl = IntPtr.Zero;
                string name = Core.world_get_env_light_by_index(i, ref hdl);
                if (hdl == IntPtr.Zero || name == null || name.Length == 0)
                    throw new Exception(Core.core_get_dll_error());
                string map = Core.world_get_env_light_map(hdl);
                if (map == null || map.Length == 0)
                    throw new Exception(Core.core_get_dll_error());
                m_models.Lights.Models.Add(new EnvmapLightModel()
                {
                    Name = name,
                    Handle = hdl,
                    Map = map
                });
            }
        }

        private void loadSceneMaterials()
        {
            // TODO
            m_models.Materials.Models.Clear();
        }

        private void loadSceneCameras()
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

        private void renderStatusChanged(object sender, PropertyChangedEventArgs args)
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
