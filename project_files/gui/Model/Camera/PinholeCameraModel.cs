using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Dll;
using gui.ViewModel.Camera;

namespace gui.Model.Camera
{
    public class PinholeCameraModel : CameraModel
    {
        public override CameraType Type => CameraType.Pinhole;

        public override CameraViewModel CreateViewModel(Models models)
        {
            return new PinholeCameraViewModel(models, this);
        }

        public float Fov
        {
            get
            {
                if(!Core.world_get_pinhole_camera_fov(Handle, out var fov))
                    throw new Exception(Core.core_get_dll_error());

                return fov;
            }
            set
            {
                if(Equals(Fov, value)) return;
                if(!Core.world_set_pinhole_camera_fov(Handle, value))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Fov));
            }
        }

        public PinholeCameraModel(IntPtr handle) : base(handle)
        {
        }
    }
}
