using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Command;
using gui.Model;
using gui.Model.Light;
using gui.Utility;
using gui.View.Light;

namespace gui.ViewModel.Light
{
    public class GoniometricLightViewModel : LightViewModel
    {
        private readonly GoniometricLightModel m_parent;

        public GoniometricLightViewModel(Models world, GoniometricLightModel parent) : base(world, parent)
        {
            m_parent = parent;
            SelectMapCommand = new SelectTextureCommand(world, () => m_parent.Map, val => m_parent.Map = val);
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(GoniometricLightModel.Position):
                    OnPropertyChanged(nameof(PositionX));
                    OnPropertyChanged(nameof(PositionY));
                    OnPropertyChanged(nameof(PositionZ));
                    break;
                case nameof(GoniometricLightModel.Map):
                    OnPropertyChanged(nameof(Map));
                    break;
            }
        }

        public override object CreateView()
        {
            return new LightView(this, new GoniometricLightView());
        }

        public string Map
        {
            get => m_parent.Map;
            set => m_parent.Map = value;
        }

        public ICommand SelectMapCommand { get; }

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
    }
}
