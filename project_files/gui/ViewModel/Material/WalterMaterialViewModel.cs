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
    public class WalterMaterialViewModel : MaterialViewModel
    {
        private readonly WalterMaterialModel m_parent;

        public WalterMaterialViewModel(WalterMaterialModel parent) : base(parent)
        {
            m_parent = parent;
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(WalterMaterialModel.Absorption):
                    OnPropertyChanged(nameof(AbsorptionX));
                    OnPropertyChanged(nameof(AbsorptionY));
                    OnPropertyChanged(nameof(AbsorptionZ));
                    break;
                case nameof(WalterMaterialModel.Roughness):
                    OnPropertyChanged(nameof(Roughness));
                    break;
                case nameof(WalterMaterialModel.RoughnessAngleX):
                    OnPropertyChanged(nameof(RoughnessAngleX));
                    break;
                case nameof(WalterMaterialModel.RoughnessAngleY):
                    OnPropertyChanged(nameof(RoughnessAngleY));
                    break;
                case nameof(WalterMaterialModel.RoughnessTex):

                    break;
            }
        }

        public override object CreateView()
        {
            throw new NotImplementedException();
        }

        public float AbsorptionX
        {
            get => m_parent.Absorption.X;
            set => m_parent.Absorption = new Vec3<float>(value, m_parent.Absorption.Y, m_parent.Absorption.Z);
        }

        public float AbsorptionY
        {
            get => m_parent.Absorption.Y;
            set => m_parent.Absorption = new Vec3<float>(m_parent.Absorption.X, value, m_parent.Absorption.Z);
        }

        public float AbsorptionZ
        {
            get => m_parent.Absorption.Z;
            set => m_parent.Absorption = new Vec3<float>(m_parent.Absorption.X, m_parent.Absorption.Y, value);
        }

        public float Roughness
        {
            get => m_parent.Roughness;
            set => m_parent.Roughness = value;
        }

        public float RoughnessAngleX
        {
            get => m_parent.RoughnessAngleX;
            set => m_parent.RoughnessAngleX = value;
        }

        public float RoughnessAngleY
        {
            get => m_parent.RoughnessAngleY;
            set => m_parent.RoughnessAngleY = value;
        }

        // TODO roughness texture
    }
}
