using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
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

            // TODO populate lights cameras and materials
        }

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

        public SynchronizedModelList<CameraModel> Cameras { get; } = new SynchronizedModelList<CameraModel>();

        public SynchronizedModelList<LightModel> Lights { get; } = new SynchronizedModelList<LightModel>();

        public SynchronizedModelList<MaterialModel> Materials { get; } = new SynchronizedModelList<MaterialModel>();

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
