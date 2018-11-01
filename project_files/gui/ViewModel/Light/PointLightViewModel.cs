using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model.Light;
using gui.Utility;

namespace gui.ViewModel.Light
{
    public class PointLightViewModel : LightViewModel
    {
        private readonly PointLightModel m_parent;

        public PointLightViewModel(PointLightModel parent) : base(parent)
        {
            m_parent = parent;
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(PointLightModel.Position):
                    OnPropertyChanged(nameof(PositionX));
                    OnPropertyChanged(nameof(PositionY));
                    OnPropertyChanged(nameof(PositionZ));
                    break;
                case nameof(PointLightModel.Intensity):
                    OnPropertyChanged(nameof(IntensityX));
                    OnPropertyChanged(nameof(IntensityY));
                    OnPropertyChanged(nameof(IntensityZ));
                    break;
            }
        }

        public override object CreateView()
        {
            throw new NotImplementedException();
        }

        public float PositionX
        {
            get => m_parent.Position.X;
            set => m_parent.Position = new Vec3<float>(value, m_parent.Position.Y, m_parent.Position.Z);
        }

        public float PositionY
        {
            get => m_parent.Position.Y;
            set => m_parent.Position = new Vec3<float>(m_parent.Position.X, value, m_parent.Position.Z);
        }

        public float PositionZ
        {
            get => m_parent.Position.Z;
            set => m_parent.Position = new Vec3<float>(m_parent.Position.X, m_parent.Position.Y, value);
        }

        public float IntensityX
        {
            get => m_parent.Intensity.X;
            set => m_parent.Intensity = new Vec3<float>(value, m_parent.Intensity.Y, m_parent.Intensity.Z);
        }

        public float IntensityY
        {
            get => m_parent.Intensity.Y;
            set => m_parent.Intensity = new Vec3<float>(m_parent.Intensity.X, value, m_parent.Intensity.Z);
        }

        public float IntensityZ
        {
            get => m_parent.Intensity.Z;
            set => m_parent.Intensity = new Vec3<float>(m_parent.Intensity.X, m_parent.Intensity.Y, value);
        }
    }
}
