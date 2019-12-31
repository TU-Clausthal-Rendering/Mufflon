using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using gui.Annotations;
using gui.Dll;
using gui.Model.Camera;
using gui.Model.Light;
using gui.Model.Material;
using gui.Utility;

namespace gui.Model.Scene
{
    /// <summary>
    /// wrapper for the core scenario
    /// </summary>
    public class ScenarioModel : INotifyPropertyChanged
    {
        private readonly WorldModel m_parent;
        public IntPtr Handle { get; }

        public ScenarioModel(WorldModel parent, IntPtr handle)
        {
            m_parent = parent;
            Handle = handle;

            // TODO populate materials
            LoadLights();
            LoadCamera();

            // add event handler if lights get added/removed during runtime
            Lights.CollectionChanged += LightsOnCollectionChanged;
        }

        private void LightsOnCollectionChanged(object sender, NotifyCollectionChangedEventArgs args)
        {
            if(args.NewItems != null)
                foreach (var lightHandle in args.NewItems)
                {
                    if(!Core.scenario_add_light(Handle, (UInt32)lightHandle))
                        throw new Exception(Core.core_get_dll_error());
                }

            if(args.OldItems != null)
                foreach (var lightHandle in args.OldItems)
                {
                    if(!Core.scenario_remove_light(Handle, (UInt32)lightHandle))
                        throw new Exception(Core.core_get_dll_error());
                }
        }

        private void LoadLights()
        {
            for (var i = 0u; i < Core.scenario_get_point_light_count(Handle); ++i)
                Lights.Add(Core.scenario_get_light_handle(Handle, (int)i, Core.LightType.Point));
            for (var i = 0u; i < Core.scenario_get_spot_light_count(Handle); ++i)
                Lights.Add(Core.scenario_get_light_handle(Handle, (int)i, Core.LightType.Spot));
            for (var i = 0u; i < Core.scenario_get_dir_light_count(Handle); ++i)
                Lights.Add(Core.scenario_get_light_handle(Handle, (int)i, Core.LightType.Directional));
            if(Core.scenario_has_envmap_light(Handle))
                Lights.Add(Core.scenario_get_light_handle(Handle, 0, Core.LightType.Envmap));
        }

        private void LoadCamera()
        {
            var handle = Core.scenario_get_camera(Handle);
            m_camera = m_parent.Cameras.Models.First((cam) => Equals(cam.Handle, handle));
        }

        public ObservableHashSet<UInt32> Lights { get; } = new ObservableHashSet<UInt32>();

        private CameraModel m_camera;

        public CameraModel Camera
        {
            get => m_camera;
            set
            {
                if(ReferenceEquals(m_camera, value)) return;

                if(!Core.scenario_set_camera(Handle, value.Handle))
                    throw new Exception(Core.core_get_dll_error());

                m_camera = value;
                OnPropertyChanged(nameof(Camera));
            }
        }

        public string Name => Core.scenario_get_name(Handle);

        public uint LodLevel
        {
            get => Core.scenario_get_global_lod_level(Handle);
            set
            {
                if(value == LodLevel) return;
                Core.scenario_set_global_lod_level(Handle, value);
                OnPropertyChanged(nameof(LodLevel));
            }
        }

        public Vec2<uint> Resolution
        {
            get
            {
                if(!Core.scenario_get_resolution(Handle, out var width, out var height))
                    throw new Exception(Core.core_get_dll_error());

                return new Vec2<uint>(width, height);
            }
            set
            {
                if(value == Resolution) return;
                if(!Core.scenario_set_resolution(Handle, value.X, value.Y))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Resolution));
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
