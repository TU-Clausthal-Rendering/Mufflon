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
    public class OrennayarMaterialViewModel : MaterialViewModel
    {
        private readonly OrennayarMaterialModel m_parent;

        public OrennayarMaterialViewModel(OrennayarMaterialModel parent) : base(parent)
        {
            m_parent = parent;
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

                    break;
                case nameof(OrennayarMaterialModel.Roughness):
                    OnPropertyChanged(nameof(Roughness));
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
    }
}
