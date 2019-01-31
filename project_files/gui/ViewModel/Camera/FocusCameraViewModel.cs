using System;
using System.CodeDom;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using gui.Model;
using gui.Model.Camera;
using gui.View.Camera;

namespace gui.ViewModel.Camera
{
    public class FocusCameraViewModel : CameraViewModel
    {
        private readonly FocusCameraModel m_parent;

        public FocusCameraViewModel(Models modelssss, FocusCameraModel parent) : base(modelssss, parent)
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
                case nameof(FocusCameraModel.SensorHeight):
                    OnPropertyChanged(nameof(SensorHeight));
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

        public float SensorHeight
        {
            get => m_parent.SensorHeight;
            set => m_parent.SensorHeight = value;
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
            return new CameraView(this, new FocusCameraView());
        }
    }
}
