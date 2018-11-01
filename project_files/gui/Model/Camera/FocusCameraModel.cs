using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.ViewModel.Camera;

namespace gui.Model.Camera
{
    public class FocusCameraModel : CameraModel
    {
        public override CameraType Type => CameraType.Focus;

        public override CameraViewModel CreateViewModel()
        {
            return new FocusCameraViewModel(this);
        }

        private float m_focalLength = 35.0f;

        public float FocalLength
        {
            get => m_focalLength;
            set
            {
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_focalLength) return;
                m_focalLength = value;
                OnPropertyChanged(nameof(FocalLength));
            }
        }

        private float m_chipHeight = 24.0f;

        public float ChipHeight
        {
            get => m_chipHeight;
            set
            {
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_chipHeight) return;
                m_chipHeight = value;
                OnPropertyChanged(nameof(ChipHeight));
            }
        }

        private float m_focusDistance;

        public float FocusDistance
        {
            get => m_focusDistance;
            set
            {
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_focusDistance) return;
                m_focusDistance = value;
                OnPropertyChanged(nameof(FocusDistance));
            }
        }

        private float m_aperture;

        public float Aperture
        {
            get => m_aperture;
            set
            {
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_aperture) return;
                m_aperture = value;
                OnPropertyChanged(nameof(Aperture));
            }
        }
    }
}
