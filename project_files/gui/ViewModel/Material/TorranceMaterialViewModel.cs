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
    public class TorranceMaterialViewModel : MaterialViewModel
    {
        private readonly TorranceMaterialModel m_parent;

        public TorranceMaterialViewModel(Models models, TorranceMaterialModel parent) : base(parent)
        {
            m_parent = parent;
            SelectAlbedoCommand = new SelectTextureCommand(models, () => m_parent.AlbedoTex, val => m_parent.AlbedoTex = val);
            SelectRoughnessCommand = new SelectTextureCommand(models, () => m_parent.RoughnessTex, val => m_parent.RoughnessTex = val);
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(TorranceMaterialModel.Albedo):
                    OnPropertyChanged(nameof(AlbedoX));
                    OnPropertyChanged(nameof(AlbedoY));
                    OnPropertyChanged(nameof(AlbedoZ));
                    break;
                case nameof(TorranceMaterialModel.AlbedoTex):
                    OnPropertyChanged(nameof(AlbedoTex));
                    break;
                case nameof(TorranceMaterialModel.UseAlbedoTexture):
                    OnPropertyChanged(nameof(AlbedoVisibility));
                    OnPropertyChanged(nameof(AlbedoTexVisibility));
                    OnPropertyChanged(nameof(SelectedAlbedo));
                    break;
                case nameof(TorranceMaterialModel.Roughness):
                    OnPropertyChanged(nameof(Roughness));
                    break;
                case nameof(TorranceMaterialModel.RoughnessAngleX):
                    OnPropertyChanged(nameof(RoughnessAngleX));
                    break;
                case nameof(TorranceMaterialModel.RoughnessAngleY):
                    OnPropertyChanged(nameof(RoughnessAngleY));
                    break;
                case nameof(TorranceMaterialModel.RoughnessAnisotropic):
                    OnPropertyChanged(nameof(RoughnessAnisotropic));
                    break;
                case nameof(TorranceMaterialModel.RoughnessTex):
                    OnPropertyChanged(nameof(RoughnessTex));
                    break;
                case nameof(TorranceMaterialModel.SelectedRoughness):
                    OnPropertyChanged(nameof(RoughnessVisibility));
                    OnPropertyChanged(nameof(RoughnessAnisotropicVisibility));
                    OnPropertyChanged(nameof(RoughnessTexVisibility));
                    OnPropertyChanged(nameof(SelectedRoughness));
                    break;
            }
        }

        public override object CreateView()
        {
            return new TorranceMaterialView(this);
        }

        public float AlbedoX
        {
            get => m_parent.Albedo.X;
            set => m_parent.Albedo = new Vec3<float>(value, m_parent.Albedo.Y, m_parent.Albedo.Z);
        }

        public float AlbedoY
        {
            get => m_parent.Albedo.Y;
            set => m_parent.Albedo = new Vec3<float>(m_parent.Albedo.X, value, m_parent.Albedo.Z);
        }

        public float AlbedoZ
        {
            get => m_parent.Albedo.Z;
            set => m_parent.Albedo = new Vec3<float>(m_parent.Albedo.X, m_parent.Albedo.Y, value);
        }

        public string AlbedoTex
        {
            get => m_parent.AlbedoTex;
            set => m_parent.AlbedoTex = value;
        }

        public ICommand SelectAlbedoCommand { get; }

        public Visibility AlbedoVisibility => m_parent.UseAlbedoTexture ? Visibility.Collapsed : Visibility.Visible;

        public Visibility AlbedoTexVisibility => m_parent.UseAlbedoTexture ? Visibility.Visible : Visibility.Collapsed;

        public int SelectedAlbedo
        {
            get => m_parent.UseAlbedoTexture ? 1 : 0;
            set
            {
                if (value == 0) m_parent.UseAlbedoTexture = false;
                if (value == 1) m_parent.UseAlbedoTexture = true;
            }
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
            get => (int) m_parent.SelectedRoughness;
            set => m_parent.SelectedRoughness = (MaterialModel.RoughnessType) Math.Max(value, 0);
        }
    }
}
