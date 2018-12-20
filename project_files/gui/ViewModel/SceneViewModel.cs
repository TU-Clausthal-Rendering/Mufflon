using System;
using System.Collections.ObjectModel;
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

            m_models.Scene.PropertyChanged += changeScene;
            m_models.Renderer.PropertyChanged += renderStatusChanged;
        }

        private void changeScene(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(SceneModel.FullPath):
                    {
                        if(m_models.Scene.FullPath != null)
                        {
                            loadSceneLights();
                            loadSceneMaterials();
                            loadSceneCameras();
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
            // Then select the proper camera out of all cameras

            //foreach(var light in m_models.Lights.Models){

            m_scenarioLoadDialog.Close();
        }

        private void loadScenarioLights(IntPtr scenarioHdl)
        {
            foreach (var light in m_models.Lights.Models)
            {
                light.IsSelected = false;
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
                        light.IsSelected = true;
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
                if (name == camera.Name)
                {
                    camera.IsSelected = true;
                }
                else
                {
                    camera.IsSelected = false;
                }
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
                    Position = new Vec3<float>(pos.x, pos.y, pos.z),
                    Intensity = new Vec3<float>(intensity.x, intensity.y, intensity.z)
                });
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
                    Position = new Vec3<float>(pos.x, pos.y, pos.z),
                    Direction = new Vec3<float>(dir.x, dir.y, dir.z),
                    Intensity = new Vec3<float>(intensity.x, intensity.y, intensity.z),
                    Width = angle,
                    FalloffStart = falloff
                });
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
                    Direction = new Vec3<float>(dir.x, dir.y, dir.z),
                    Radiance = new Vec3<float>(radiance.x, radiance.y, radiance.z)
                });
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
