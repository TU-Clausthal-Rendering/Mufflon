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
                m_models.World.CurrentScenario = m_selectedScenario.Cargo;
            }
        }
        public ObservableCollection<ComboBoxItem<ScenarioModel>> Scenarios { get; } = new ObservableCollection<ComboBoxItem<ScenarioModel>>();
        public bool CanLoadLastScenes => LastScenes.Count > 0 && !m_models.Renderer.IsRendering;

        public SceneViewModel(Models models)
        {
            m_models = models;

            foreach (string path in m_models.Settings.LastWorlds)
            {
                LastScenes.Add(new SceneMenuItem(m_models)
                {
                    Filename = Path.GetFileName(path),
                    Path = path
                });
            }

            m_models.Settings.LastWorlds.CollectionChanged += LastWorldsOnCollectionChanged;

            // Register the handle for scene changes
            // TODO remove after relocation materials and camera
            m_models.Materials.Models.CollectionChanged += OnMaterialsCollectionChanged;
            m_models.Cameras.Models.CollectionChanged += OnCamerasCollectionChanged;

            // assume no scene loaded
            Debug.Assert(m_models.World == null);
            m_models.PropertyChanged += ModelsOnPropertyChanged;
            m_models.Renderer.PropertyChanged += OnRendererChange;
        }

        private void LastWorldsOnCollectionChanged(object sender, NotifyCollectionChangedEventArgs args)
        {
            // TODO make more efficient by using args
            LastScenes.Clear();
            foreach (string path in m_models.Settings.LastWorlds)
            {
                LastScenes.Add(new SceneMenuItem(m_models)
                {
                    Filename = Path.GetFileName(path),
                    Path = path
                });
            }
            OnPropertyChanged(nameof(CanLoadLastScenes));
        }

        private void ModelsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.World):
                    // subscribe to new model
                    if (m_models.World != null)
                    {
                        m_models.World.PropertyChanged += OnSceneChanged;
                        m_models.World.Scenarios.CollectionChanged += ScenariosOnCollectionChanged;
                        if (m_models.World.FullPath != null)
                        {
                            // Temporarily disable handlers
                            // TODO remove after relocation materials and camera
                            m_models.Materials.Models.CollectionChanged -= OnMaterialsCollectionChanged;
                            m_models.Cameras.Models.CollectionChanged -= OnCamerasCollectionChanged;
                            LoadSceneMaterials();
                            LoadSceneCameras();
                            m_models.Materials.Models.CollectionChanged += OnMaterialsCollectionChanged;
                            m_models.Cameras.Models.CollectionChanged += OnCamerasCollectionChanged;
                        }
                    }

                    // refresh views
                    MakeScenerioViews();

                    break;
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
                    }
                }
            }
        }

        private void OnMaterialChanged(object sender, PropertyChangedEventArgs args)
        {
            MaterialModel material = (sender as MaterialModel);
            // TODO
        }

        private void OnCameraChanged(object sender, PropertyChangedEventArgs args)
        {
            // TODO: we can find out what was changed easily
            CameraModel camera = (sender as CameraModel);
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
        }

        private void OnSceneChanged(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(WorldModel.CurrentScenario):
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

            if (m_models.World == null)
            {
                SelectedScenario = null;
                return;
            }

            // code assumes that an active scenario exists
            Debug.Assert(m_models.World.CurrentScenario != null);

            foreach (var scenario in m_models.World.Scenarios)
            {
                // add scenario view
                var view = new ComboBoxItem<ScenarioModel>(scenario.Name, scenario);
                Scenarios.Add(view);
                // set selected item
                if (ReferenceEquals(scenario, m_models.World.CurrentScenario))
                    SelectedScenario = view;
            }
            LoadScenarioViews();
        }

        // TODO remove this
        private void LoadScenarioViews()
        {
            // First fetch the scenario-specific information
            loadScenarioMaterials(m_models.World.CurrentScenario.Handle);
            LoadScenarioCamera(m_models.World.CurrentScenario.Handle);

            // Change display and renderer resolution
            m_models.Viewport.RenderWidth = (int)m_models.World.CurrentScenario.Resolution.X;
            m_models.Viewport.RenderHeight = (int)m_models.World.CurrentScenario.Resolution.Y;
            //m_scenarioLoadDialog.Close();
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
