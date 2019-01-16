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
                foreach (var lightName in args.NewItems)
                {
                    if(!Core.scenario_add_light(Handle, lightName as string))
                        throw new Exception(Core.core_get_dll_error());
                }

            if(args.OldItems != null)
                foreach (var lightName in args.OldItems)
                {
                    if(!Core.scenario_remove_light(Handle, lightName as string))
                        throw new Exception(Core.core_get_dll_error());
                }
        }

        private void LoadLights()
        {
            var numPointLights = Core.scenario_get_point_light_count(Handle);
            for (var i = 0u; i < numPointLights; ++i)
            {
                var name = Core.scenario_get_point_light_name(Handle, i);
                if (string.IsNullOrEmpty(name))
                    throw new Exception(Core.core_get_dll_error());

                Lights.Add(name);
            }

            var numSpotLights = Core.scenario_get_spot_light_count(Handle);
            for (var i = 0u; i < numSpotLights; ++i)
            {
                var name = Core.scenario_get_spot_light_name(Handle, i);
                if (string.IsNullOrEmpty(name))
                    throw new Exception(Core.core_get_dll_error());

                Lights.Add(name);
            }


            var numDirLights = Core.scenario_get_dir_light_count(Handle);
            for (var i = 0u; i < numDirLights; ++i)
            {
                var name = Core.scenario_get_dir_light_name(Handle, i);
                if (string.IsNullOrEmpty(name))
                    throw new Exception(Core.core_get_dll_error());

                Lights.Add(name);
            }

            // TODO make sure that a second envmap cannot be added later on
            if (Core.scenario_has_envmap_light(Handle))
            {
                var name = Core.scenario_get_envmap_light_name(Handle);
                if (string.IsNullOrEmpty(name))
                    throw new Exception(Core.core_get_dll_error());

                Lights.Add(name);
            }
        }

        public ObservableHashSet<string> Lights { get; } = new ObservableHashSet<string>();

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
