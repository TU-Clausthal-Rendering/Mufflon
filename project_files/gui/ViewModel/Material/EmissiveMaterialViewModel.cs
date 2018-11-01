using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model.Material;
using gui.Utility;

namespace gui.ViewModel.Material
{
    public class EmissiveMaterialViewModel : MaterialViewModel
    {
        private readonly EmissiveMaterialModel m_parent;

        public EmissiveMaterialViewModel(EmissiveMaterialModel parent) : base(parent)
        {
            m_parent = parent;
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(EmissiveMaterialModel.Radiance):
                    OnPropertyChanged(nameof(RadianceX));
                    OnPropertyChanged(nameof(RadianceY));
                    OnPropertyChanged(nameof(RadianceZ));
                    break;
                case nameof(EmissiveMaterialModel.RadianceTex):

                    break;
                case nameof(EmissiveMaterialModel.Scale):
                    OnPropertyChanged(nameof(Scale));
                    break;
            }
        }

        public override object CreateView()
        {
            throw new NotImplementedException();
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

        public float Scale
        {
            get => m_parent.Scale;
            set => m_parent.Scale = value;
        }

        // TODO radiance tex
    }
}
