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
    public class DirectionalLightViewModel : LightViewModel
    {
        private readonly DirectionalLightModel m_parent;

        public DirectionalLightViewModel(Models world, DirectionalLightModel parent) : base(world, parent)
        {
            m_parent = parent;
            world.World.PropertyChanged += ModelOnPropertyChanged;
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
                case nameof(DirectionalLightModel.Irradiance):
                    OnPropertyChanged(nameof(RadianceX));
                    OnPropertyChanged(nameof(RadianceY));
                    OnPropertyChanged(nameof(RadianceZ));
                    break;
                case nameof(Models.World.AnimationFrameCurrent):
                    OnPropertyChanged(nameof(DirectionX));
                    OnPropertyChanged(nameof(DirectionY));
                    OnPropertyChanged(nameof(DirectionZ));
                    OnPropertyChanged(nameof(RadianceX));
                    OnPropertyChanged(nameof(RadianceY));
                    OnPropertyChanged(nameof(RadianceZ));
                    break;
            }
        }

        public override object CreateView()
        {
            return new LightView(this, new DirectionalLightView());
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
            get => m_parent.Irradiance.X;
            set => m_parent.Irradiance = new Vec3<float>(value, m_parent.Irradiance.Y, m_parent.Irradiance.Z);
        }

        public float RadianceY
        {
            get => m_parent.Irradiance.Y;
            set => m_parent.Irradiance = new Vec3<float>(m_parent.Irradiance.X, value, m_parent.Irradiance.Z);
        }

        public float RadianceZ
        {
            get => m_parent.Irradiance.Z;
            set => m_parent.Irradiance = new Vec3<float>(m_parent.Irradiance.X, m_parent.Irradiance.Y, value);
        }
    }
}
