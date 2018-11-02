using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using gui.Command;
using gui.Model;
using gui.Model.Material;
using gui.Utility;
using gui.View.Material;

namespace gui.ViewModel.Material
{
    public class WalterMaterialViewModel : MaterialViewModel
    {
        private readonly WalterMaterialModel m_parent;

        public WalterMaterialViewModel(Models models, WalterMaterialModel parent) : base(parent)
        {
            m_parent = parent;
            SelectRoughnessCommand = new SelectTextureCommand(models, () => m_parent.RoughnessTex, val => m_parent.RoughnessTex = val);
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
                case nameof(WalterMaterialModel.RoughnessAnisotropic):
                    OnPropertyChanged(nameof(RoughnessAnisotropic));
                    break;
                case nameof(WalterMaterialModel.RoughnessTex):
                    OnPropertyChanged(nameof(RoughnessTex));
                    break;
                case nameof(WalterMaterialModel.SelectedRoughness):
                    OnPropertyChanged(nameof(RoughnessVisibility));
                    OnPropertyChanged(nameof(RoughnessAnisotropicVisibility));
                    OnPropertyChanged(nameof(RoughnessTexVisibility));
                    OnPropertyChanged(nameof(SelectedRoughness));
                    break;
            }
        }

        public override object CreateView()
        {
            return new WalterMaterialView(this);
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

        public float RoughnessAnisotropic
        {
            get => m_parent.RoughnessAnisotropic;
            set => m_parent.RoughnessAnisotropic = value;
        }

        public string RoughnessTex
        {
            get => m_parent.RoughnessTex;
            set => m_parent.RoughnessTex = value;
        }

        public ICommand SelectRoughnessCommand { get; }

        public Visibility RoughnessVisibility => m_parent.SelectedRoughness == MaterialModel.RoughnessType.Isotropic
            ? Visibility.Visible
            : Visibility.Collapsed;

        public Visibility RoughnessAnisotropicVisibility => m_parent.SelectedRoughness == MaterialModel.RoughnessType.Anisotropic
            ? Visibility.Visible
            : Visibility.Collapsed;

        public Visibility RoughnessTexVisibility => m_parent.SelectedRoughness == MaterialModel.RoughnessType.Texture
            ? Visibility.Visible
            : Visibility.Collapsed;

        public int SelectedRoughness
        {
            get => (int)m_parent.SelectedRoughness;
            set => m_parent.SelectedRoughness = (MaterialModel.RoughnessType)Math.Max(value, 0);
        }
    }
}
