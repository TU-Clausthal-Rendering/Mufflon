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
    public class TorranceMaterialViewModel : MaterialViewModel
    {
        private readonly TorranceMaterialModel m_parent;

        public TorranceMaterialViewModel(TorranceMaterialModel parent) : base(parent)
        {
            m_parent = parent;
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
                case nameof(TorranceMaterialModel.RoughnessTex):

                    break;
            }
        }

        public override object CreateView()
        {
            throw new NotImplementedException();
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

        // TODO albedo tex, roughness tex
    }
}
