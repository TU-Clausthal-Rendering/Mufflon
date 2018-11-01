using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model.Light;
using gui.Utility;
using gui.View.Light;

namespace gui.ViewModel.Light
{
    public class DirectionalLightViewModel : LightViewModel
    {
        private readonly DirectionalLightModel m_parent;

        public DirectionalLightViewModel(DirectionalLightModel parent) : base(parent)
        {
            m_parent = parent;
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(DirectionalLightModel.Direction):
                    OnPropertyChanged(nameof(DirectionX));
                    OnPropertyChanged(nameof(DirectionY));
                    OnPropertyChanged(nameof(DirectionZ));
                    break;
                case nameof(DirectionalLightModel.Radiance):
                    OnPropertyChanged(nameof(RadianceX));
                    OnPropertyChanged(nameof(RadianceY));
                    OnPropertyChanged(nameof(RadianceZ));
                    break;
            }
        }

        public override object CreateView()
        {
            return new DirectionalLightView(this);
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

        public float RadianceX
        {
            get => m_parent.Radiance.X;
            set => m_parent.Radiance = new Vec3<float>(value, m_parent.Radiance.Y, m_parent.Radiance.Z);
        }

        public float RadianceY
        {
            get => m_parent.Radiance.Y;
            set => m_parent.Radiance = new Vec3<float>(m_parent.Radiance.X, value, m_parent.Radiance.Z);
        }

        public float RadianceZ
        {
            get => m_parent.Radiance.Z;
            set => m_parent.Radiance = new Vec3<float>(m_parent.Radiance.X, m_parent.Radiance.Y, value);
        }
    }
}
