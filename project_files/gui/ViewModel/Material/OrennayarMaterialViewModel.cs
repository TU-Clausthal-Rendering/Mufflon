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
    public class OrennayarMaterialViewModel : MaterialViewModel
    {
        private readonly OrennayarMaterialModel m_parent;

        public OrennayarMaterialViewModel(Models models ,OrennayarMaterialModel parent) : base(models, parent)
        {
            m_parent = parent;
            SelectAlbedoCommand = new SelectTextureCommand(models, () => m_parent.AlbedoTex, val => m_parent.AlbedoTex = val);
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(OrennayarMaterialModel.Albedo):
                    OnPropertyChanged(nameof(AlbedoX));
                    OnPropertyChanged(nameof(AlbedoY));
                    OnPropertyChanged(nameof(AlbedoZ));
                    break;
                case nameof(OrennayarMaterialModel.AlbedoTex):
                    OnPropertyChanged(nameof(AlbedoTex));
                    break;
                case nameof(OrennayarMaterialModel.UseAlbedoTexture):
                    OnPropertyChanged(nameof(AlbedoVisibility));
                    OnPropertyChanged(nameof(AlbedoTexVisibility));
                    OnPropertyChanged(nameof(SelectedAlbedo));
                    break;
                case nameof(OrennayarMaterialModel.Roughness):
                    OnPropertyChanged(nameof(Roughness));
                    break;
            }
        }

        protected override UIElement CreateInternalView()
        {
            return new OrennayarMaterialView();
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
    }
}
