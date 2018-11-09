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
    public class EmissiveMaterialViewModel : MaterialViewModel
    {
        private readonly EmissiveMaterialModel m_parent;

        public EmissiveMaterialViewModel(Models models, EmissiveMaterialModel parent) : base(models, parent)
        {
            m_parent = parent;
            SelectRadianceCommand = new SelectTextureCommand(models, () => m_parent.RadianceTex, val => m_parent.RadianceTex = val);
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
                    OnPropertyChanged(nameof(RadianceTex));
                    break;
                case nameof(EmissiveMaterialModel.UseRadianceTexture):
                    OnPropertyChanged(nameof(RadianceVisibility));
                    OnPropertyChanged(nameof(RadianceTexVisibility));
                    OnPropertyChanged(nameof(SelectedRadiance));
                    break;
                case nameof(EmissiveMaterialModel.Scale):
                    OnPropertyChanged(nameof(Scale));
                    break;
            }
        }

        protected override UIElement CreateInternalView()
        {
            return new EmissiveMaterialView();
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

        public string RadianceTex
        {
            get => m_parent.RadianceTex;
            set => m_parent.RadianceTex = value;
        }

        public ICommand SelectRadianceCommand { get; }

        public Visibility RadianceVisibility => m_parent.UseRadianceTexture ? Visibility.Collapsed : Visibility.Visible;

        public Visibility RadianceTexVisibility => m_parent.UseRadianceTexture ? Visibility.Visible : Visibility.Collapsed;

        public int SelectedRadiance
        {
            get => m_parent.UseRadianceTexture ? 1 : 0;
            set
            {
                if (value == 0) m_parent.UseRadianceTexture = false;
                if (value == 1) m_parent.UseRadianceTexture = true;
            }
        }

        public float Scale
        {
            get => m_parent.Scale;
            set => m_parent.Scale = value;
        }
    }
}
