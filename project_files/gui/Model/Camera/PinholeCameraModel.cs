using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.ViewModel.Camera;

namespace gui.Model.Camera
{
    public class PinholeCameraModel : CameraModel
    {
        public override CameraType Type => CameraType.Pinhole;

        public override CameraViewModel CreateViewModel()
        {
            return new PinholeCameraViewModel(this);
        }

        private float m_fov = 25.0f;

        public float Fov
        {
            get => m_fov;
            set
            {
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_fov) return;
                m_fov = value;
                OnPropertyChanged(nameof(Fov));
            }
        }
    }
}
