using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using gui.Annotations;
using gui.Dll;
using gui.Utility;
using gui.ViewModel;
using gui.ViewModel.Camera;

namespace gui.Model.Camera
{
    /// <summary>
    /// Base class for camera models
    /// </summary>
    public abstract class CameraModel : INotifyPropertyChanged
    {
        private Vec3<float> m_originalPosition;
        private Vec3<float> m_originalViewDirection;
        private Vec3<float> m_originalUp;

        public enum CameraType
        {
            Pinhole,
            Focus,
            Ortho
        }

        protected CameraModel(IntPtr handle)
        {
            Handle = handle;
            m_originalPosition = Position;
            m_originalViewDirection = ViewDirection;
            m_originalUp = Up;
        }

        public abstract CameraType Type { get; }

        public string Name => Core.world_get_camera_name(Handle);

        public Vec3<float> Position
        {
            get
            { 
                if(!Core.world_get_camera_position(Handle, out var pos))
                    throw new Exception(Core.core_get_dll_error());

                return pos.ToUtilityVec();
            }
            set
            {
                if (Equals(Position, value)) return;
                if(!Core.world_set_camera_position(Handle, new Core.Vec3(value)))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Position));
            }
        }

        public Vec3<float> ViewDirection
        {
            get
            {
                if(!Core.world_get_camera_direction(Handle, out var dir))
                    throw new Exception(Core.core_get_dll_error());

                return dir.ToUtilityVec();
            }
            set
            {
                if (Equals(ViewDirection, value)) return;
                if(!Core.world_set_camera_direction(Handle, new Core.Vec3(value), new Core.Vec3(Up)))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(ViewDirection));
            }
        }

        public Vec3<float> Up
        {
            get
            {
                if (!Core.world_get_camera_up(Handle, out var up))
                    throw new Exception(Core.core_get_dll_error());

                return up.ToUtilityVec();
            }
            set
            {
                if (Equals(Up, value)) return;
                if (!Core.world_set_camera_direction(Handle, new Core.Vec3(ViewDirection), new Core.Vec3(value)))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Up));
            }
        }

        public float Near
        {
            get
            {
                if (!Core.world_get_camera_near(Handle, out var up))
                    throw new Exception(Core.core_get_dll_error());

                return up;
            }
            set
            {
                if (Equals(Near, value)) return;
                if (!Core.world_set_camera_near(Handle, value))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Near));
            }
        }

        public float Far
        {
            get
            {
                if (!Core.world_get_camera_far(Handle, out var far))
                    throw new Exception(Core.core_get_dll_error());

                return far;
            }
            set
            {
                if (Equals(Far, value)) return;
                if (!Core.world_set_camera_far(Handle, value))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Far));
            }
        }

        public IntPtr Handle { get; }

        // Resets the camera's rotation and translation to how it was upon loading
        public void ResetTransRot()
        {
            Position = m_originalPosition;
            ViewDirection = m_originalViewDirection;
            Up = m_originalUp;
            OnPropertyChanged(nameof(Position));
            OnPropertyChanged(nameof(ViewDirection));
            OnPropertyChanged(nameof(Up));
        }

        /// <summary>
        /// creates a new view model based on this model
        /// </summary>
        /// <param name="models"></param>
        /// <returns></returns>
        public abstract CameraViewModel CreateViewModel(Models models);

        public static CameraModel MakeFromHandle(IntPtr handle, Core.CameraType type)
        {
            switch (type)
            {
                case Core.CameraType.Pinhole:
                    return new PinholeCameraModel(handle);
                case Core.CameraType.Focus:
                    return new FocusCameraModel(handle);
                default:
                    // not implemented
                    Debug.Assert(false);
                    throw new NotImplementedException();
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
