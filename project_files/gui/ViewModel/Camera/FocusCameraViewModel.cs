using System;
using System.CodeDom;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using gui.Model.Camera;
using gui.View.Camera;

namespace gui.ViewModel.Camera
{
    public class FocusCameraViewModel : CameraViewModel
    {
        private readonly FocusCameraModel m_parent;

        public FocusCameraViewModel(FocusCameraModel parent) : base(parent)
        {
            m_parent = parent;
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(FocusCameraModel.FocalLength):
                    OnPropertyChanged(nameof(FocalLength));
                    break;
                case nameof(FocusCameraModel.ChipHeight):
                    OnPropertyChanged(nameof(ChipHeight));
                    break;
                case nameof(FocusCameraModel.FocusDistance):
                    OnPropertyChanged(nameof(FocusDistance));
                    break;
                case nameof(FocusCameraModel.Aperture):
                    OnPropertyChanged(nameof(Aperture));
                    break;
            }
        }

        public float FocalLength
        {
            get => m_parent.FocalLength;
            set => m_parent.FocalLength = value;
        }

        public float ChipHeight
        {
            get => m_parent.ChipHeight;
            set => m_parent.ChipHeight = value;
        }

        public float FocusDistance
        {
            get => m_parent.FocusDistance;
            set => m_parent.FocusDistance = value;
        }

        public float Aperture
        {
            get => m_parent.Aperture;
            set => m_parent.Aperture = value;
        }

        public override object CreateView()
        {
            return new FocusCameraView(this);
        }
    }
}
