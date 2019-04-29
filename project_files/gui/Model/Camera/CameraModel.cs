using System;
using System.Collections.Generic;
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
        private struct OriginalData
        {
            public Vec3<float> position;
            public Vec3<float> viewDirection;
            public Vec3<float> up;
        }
        private readonly float m_originalNear;
        private readonly float m_originalFar;

        private readonly IList<OriginalData> m_originalData = new List<OriginalData>();

        public enum CameraType
        {
            Pinhole,
            Focus,
            Ortho
        }

        protected CameraModel(IntPtr handle)
        {
            Handle = handle;
            // We need to loop over all frames to get the camera positions for each one
            uint frameStart, frameEnd;
            if(!Core.world_get_frame_start(out frameStart))
                throw new Exception(Core.core_get_dll_error());
            if(!Core.world_get_frame_end(out frameEnd))
                throw new Exception(Core.core_get_dll_error());

            for (uint frame = frameStart; frame <= frameEnd; ++frame)
            {
                Core.Vec3 position, direction, up;
                if (!(Core.world_get_camera_position(handle, out position, frame - frameStart)
                    && Core.world_get_camera_direction(handle, out direction, frame - frameStart)
                    && Core.world_get_camera_up(handle, out up, frame - frameStart)))
                    throw new Exception(Core.core_get_dll_error());
                m_originalData.Add(new OriginalData {
                    position = Position,
                    viewDirection = ViewDirection,
                    up = Up
                });
            }
            m_originalNear = Near;
            m_originalFar = Far;
        }

        public abstract CameraType Type { get; }

        public string Name => Core.world_get_camera_name(Handle);

        public Vec3<float> Position
        {
            get
            { 
                if(!Core.world_get_camera_current_position(Handle, out var pos))
                    throw new Exception(Core.core_get_dll_error());

                return pos.ToUtilityVec();
            }
            set
            {
                if (Equals(Position, value)) return;
                if(!Core.world_set_camera_current_position(Handle, new Core.Vec3(value)))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(Position));
            }
        }

        public Vec3<float> ViewDirection
        {
            get
            {
                if(!Core.world_get_camera_current_direction(Handle, out var dir))
                    throw new Exception(Core.core_get_dll_error());

                return dir.ToUtilityVec();
            }
            set
            {
                if (Equals(ViewDirection, value)) return;
                if(!Core.world_set_camera_current_direction(Handle, new Core.Vec3(value), new Core.Vec3(Up)))
                    throw new Exception(Core.core_get_dll_error());

                OnPropertyChanged(nameof(ViewDirection));
            }
        }

        public Vec3<float> Up
        {
            get
            {
                if (!Core.world_get_camera_current_up(Handle, out var up))
                    throw new Exception(Core.core_get_dll_error());

                return up.ToUtilityVec();
            }
            set
            {
                if (Equals(Up, value)) return;
                if (!Core.world_set_camera_current_direction(Handle, new Core.Vec3(ViewDirection), new Core.Vec3(value)))
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

        public uint PathSegments
        {
            get
            {
                uint segments;
                if (!Core.world_get_camera_path_segment_count(Handle, out segments))
                    throw new Exception(Core.core_get_dll_error());
                return segments;
            }
        }

        public void Move(Vec3<float> offset)
        {
            if (!Core.scene_move_active_camera(offset.X, offset.Y, offset.Z))
                throw new Exception(Core.core_get_dll_error());
            OnPropertyChanged(nameof(Position));
        }

        public void Rotate(Vec2<float> rotation)
        {
            if (!Core.scene_rotate_active_camera(rotation.X, rotation.Y, 0))
                throw new Exception(Core.core_get_dll_error());
            OnPropertyChanged(nameof(ViewDirection));
            OnPropertyChanged(nameof(Up));
        }

        // Resets the camera's rotation and translation to how it was upon loading
        public void Reset(uint frame)
        {
            Position = m_originalData[(int)frame].position;
            ViewDirection = m_originalData[(int)frame].viewDirection;
            Up = m_originalData[(int)frame].up;
            Near = m_originalNear;
            Far = m_originalFar;
            ResetConcreteModel();
            OnPropertyChanged(nameof(Position));
            OnPropertyChanged(nameof(ViewDirection));
            OnPropertyChanged(nameof(Up));
        }

        protected abstract void ResetConcreteModel();

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
