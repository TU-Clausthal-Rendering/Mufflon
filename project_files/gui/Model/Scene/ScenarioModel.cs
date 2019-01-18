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
        public IntPtr Handle { get; }

        public ScenarioModel(IntPtr handle)
        {
            Handle = handle;

            // TODO populate cameras and materials
            LoadLights();

            // add event handler if lights get added/removed during runtime
            Lights.CollectionChanged += LightsOnCollectionChanged;
        }

        private void LightsOnCollectionChanged(object sender, NotifyCollectionChangedEventArgs args)
        {
            if(args.NewItems != null)
                foreach (var lightHandle in args.NewItems)
                {
                    if(!Core.scenario_add_light(Handle, (IntPtr)lightHandle))
                        throw new Exception(Core.core_get_dll_error());
                }

            if(args.OldItems != null)
                foreach (var lightHandle in args.OldItems)
                {
                    if(!Core.scenario_remove_light(Handle, (IntPtr)lightHandle))
                        throw new Exception(Core.core_get_dll_error());
                }
        }

        private void LoadLights()
        {
            for (var i = 0u; i < Core.scenario_get_point_light_count(Handle); ++i)
                Lights.Add(Core.scenario_get_light_handle(Handle, i, Core.LightType.POINT));
            for (var i = 0u; i < Core.scenario_get_spot_light_count(Handle); ++i)
                Lights.Add(Core.scenario_get_light_handle(Handle, i, Core.LightType.SPOT));
            for (var i = 0u; i < Core.scenario_get_dir_light_count(Handle); ++i)
                Lights.Add(Core.scenario_get_light_handle(Handle, i, Core.LightType.DIRECTIONAL));
            if(Core.scenario_has_envmap_light(Handle))
                Lights.Add(Core.scenario_get_light_handle(Handle, 0u, Core.LightType.ENVMAP));
        }

        public ObservableHashSet<IntPtr> Lights { get; } = new ObservableHashSet<IntPtr>();

        public string Name => Core.scenario_get_name(Handle);

        public ulong LodLevel
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
