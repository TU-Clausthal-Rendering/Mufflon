using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.ViewModel.Camera;

namespace gui.Model.Camera
{
    public class OrthoCameraModel : CameraModel
    {
        public override CameraType Type => CameraType.Ortho;

        public override CameraViewModel CreateViewModel()
        {
            throw new NotImplementedException();
        }

        private float m_width;

        public float Width
        {
            get => m_width;
            set
            {
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_width) return;
                m_width = value;
                OnPropertyChanged(nameof(Width));
            }
        }

        private float m_height;

        public float Height
        {
            get => m_height;
            set
            {
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_height) return;
                m_height = value;
                OnPropertyChanged(nameof(Height));
            }
        }
    }
}
