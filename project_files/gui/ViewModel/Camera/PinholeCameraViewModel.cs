using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model;
using gui.Model.Camera;
using gui.View.Camera;

namespace gui.ViewModel.Camera
{
    public class PinholeCameraViewModel : CameraViewModel
    {
        private readonly PinholeCameraModel m_parent;

        public PinholeCameraViewModel(Models modelssss, PinholeCameraModel parent) : base(modelssss, parent)
        {
            m_parent = parent;
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(PinholeCameraModel.Fov):
                    OnPropertyChanged(nameof(Fov));
                    break;
            }
        }

        public float Fov
        {
            get => m_parent.Fov;
            set => m_parent.Fov = value;
        }

        public override object CreateView()
        {
            return new CameraView(this, new PinholeCameraView());
        }
    }
}
