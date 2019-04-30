using System;
using gui.Dll;
using gui.ViewModel.Camera;

namespace gui.Model.Camera
{
    public class FocusCameraModel : CameraModel
    {
        private readonly float m_originalFocalLength;
        private readonly float m_originalSensorHeight;
        private readonly float m_originalFocusDistance;
        private readonly float m_originalAperture;

        public override CameraType Type => CameraType.Focus;

        public override CameraViewModel CreateViewModel(Models models)
        {
            return new FocusCameraViewModel(models, this);
        }

        public float FocalLength
        {
            get
            {
                if(!Core.world_get_focus_camera_focal_length(Handle, out var res))
                    throw new Exception(Core.core_get_dll_error());

                return res;
            }
            set
            {
                if(Equals(FocalLength, value)) return;
                if(!Core.world_set_focus_camera_focal_length(Handle, value))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(FocalLength));
            }
        }

        public float SensorHeight
        {
            get
            {
                if (!Core.world_get_focus_camera_sensor_height(Handle, out var res))
                    throw new Exception(Core.core_get_dll_error());

                return res;
            }
            set
            {
                if (Equals(SensorHeight, value)) return;
                if (!Core.world_set_focus_camera_sensor_height(Handle, value))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(SensorHeight));
            }
        }

        public float FocusDistance
        {
            get
            {
                if (!Core.world_get_focus_camera_focus_distance(Handle, out var res))
                    throw new Exception(Core.core_get_dll_error());

                return res;
            }
            set
            {
                if (Equals(FocusDistance, value)) return;
                if (!Core.world_set_focus_camera_focus_distance(Handle, value))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(FocusDistance));
            }
        }

        public float Aperture
        {
            get
            {
                if (!Core.world_get_focus_camera_aperture(Handle, out var res))
                    throw new Exception(Core.core_get_dll_error());

                return res;
            }
            set
            {
                if (Equals(FocalLength, value)) return;
                if (!Core.world_set_focus_camera_aperture(Handle, value))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Aperture));
            }
        }

        public FocusCameraModel(IntPtr handle) : base(handle)
        {
            m_originalFocalLength = FocalLength;
            m_originalSensorHeight = SensorHeight;
            m_originalFocusDistance = FocusDistance;
            m_originalAperture = Aperture;
        }

        protected override void ResetConcreteModel()
        {
            FocalLength = m_originalFocalLength;
            SensorHeight = m_originalSensorHeight;
            FocusDistance = m_originalFocusDistance;
            Aperture = m_originalAperture;
            OnPropertyChanged(nameof(FocalLength));
            OnPropertyChanged(nameof(SensorHeight));
            OnPropertyChanged(nameof(FocusDistance));
            OnPropertyChanged(nameof(Aperture));
        }
    }
}
