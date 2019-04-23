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
        private readonly float m_originalWidth;
        private readonly float m_originalHeight;

        public override CameraType Type => CameraType.Ortho;

        public override CameraViewModel CreateViewModel(Models models)
        {
            return new OrthoCameraViewModel(models, this);
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

        public OrthoCameraModel(IntPtr handle) : base(handle)
        {
            m_originalWidth = Width;
            m_originalHeight = Height;
        }


        protected override void ResetConcreteModel()
        {
            Width = m_originalWidth;
            Height = m_originalHeight;
            OnPropertyChanged(nameof(Width));
            OnPropertyChanged(nameof(Height));
        }
    }
}
