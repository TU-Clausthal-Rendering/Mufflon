using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model;
using gui.Model.Light;
using gui.Utility;
using gui.View.Light;

namespace gui.ViewModel.Light
{
    public class SpotLightViewModel : LightViewModel
    {
        private readonly SpotLightModel m_parent;

        public SpotLightViewModel(Models world, SpotLightModel parent) : base(world, parent)
        {
            m_parent = parent;
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(SpotLightModel.Position):
                    OnPropertyChanged(nameof(PositionX));
                    OnPropertyChanged(nameof(PositionY));
                    OnPropertyChanged(nameof(PositionZ));
                    break;
                case nameof(SpotLightModel.Direction):
                    OnPropertyChanged(nameof(DirectionX));
                    OnPropertyChanged(nameof(DirectionY));
                    OnPropertyChanged(nameof(DirectionZ));
                    break;
                case nameof(SpotLightModel.Intensity):
                    OnPropertyChanged(nameof(IntensityX));
                    OnPropertyChanged(nameof(IntensityY));
                    OnPropertyChanged(nameof(IntensityZ));
                    break;
                case nameof(SpotLightModel.Width):
                    OnPropertyChanged(nameof(Width));
                    break;
                case nameof(SpotLightModel.Falloff):
                    OnPropertyChanged(nameof(FalloffStart));
                    break;
            }
        }

        public override object CreateView()
        {
            return new LightView(this, new SpotLightView());
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

        public float DirectionX
        {
            get => m_parent.Direction.X;
            set => m_parent.Direction = new Vec3<float>(value, m_parent.Direction.Y, m_parent.Direction.Z);
        }

        public float DirectionY
        {
            get => m_parent.Direction.Y;
            set => m_parent.Direction = new Vec3<float>(m_parent.Direction.X, value, m_parent.Direction.Z);
        }

        public float DirectionZ
        {
            get => m_parent.Direction.Z;
            set => m_parent.Direction = new Vec3<float>(m_parent.Direction.X, m_parent.Direction.Y, value);
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

        public float Width
        {
            get => m_parent.Width;
            set => m_parent.Width = value;
        }

        public float FalloffStart
        {
            get => m_parent.Falloff;
            set => m_parent.Falloff = value;
        }
    }
}
