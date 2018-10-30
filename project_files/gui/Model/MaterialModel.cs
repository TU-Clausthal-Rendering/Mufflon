using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using gui.Annotations;
using gui.Utility;

namespace gui.Model
{
    public class MaterialModel : INotifyPropertyChanged
    {
        public enum MaterialType
        {
            Lambert,
            Torrance,
            Walter,
            Emmisive,
            Orennayar,
            Blend,
            Fresnel
        }

        public MaterialModel(MaterialType type)
        {
            Type = type;
        }

        public MaterialType Type { get; }

        private Vec3<float> m_albedo = new Vec3<float>(0.5f);

        public Vec3<float> Albedo
        {
            get
            {
                Debug.Assert(Type == MaterialType.Lambert || Type == MaterialType.Torrance
                    || Type == MaterialType.Orennayar);
                return m_albedo;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Lambert || Type == MaterialType.Torrance
                             || Type == MaterialType.Orennayar);
                if (Equals(value, m_albedo)) return;
                m_albedo = value;
                OnPropertyChanged(nameof(Albedo));
            }
        }

        private string m_albedoTex = String.Empty;

        public string AlbedoTex
        {
            get
            {
                Debug.Assert(Type == MaterialType.Lambert || Type == MaterialType.Torrance
                             || Type == MaterialType.Orennayar);
                return m_albedoTex;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Lambert || Type == MaterialType.Torrance
                             || Type == MaterialType.Orennayar);
                if (Equals(value, m_albedoTex)) return;
                m_albedoTex = value;
                OnPropertyChanged(nameof(AlbedoTex));
            }
        }

        private float m_roughness = 0.5f;
        // isotrophic roughness value
        public float Roughness
        {
            get
            {
                Debug.Assert(Type == MaterialType.Torrance || Type == MaterialType.Walter
                    || Type == MaterialType.Orennayar);
                return m_roughness;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Torrance || Type == MaterialType.Walter
                             || Type == MaterialType.Orennayar);
                if (Equals(value, m_roughness)) return;
                m_roughness = value;
                OnPropertyChanged(nameof(Roughness));
            }
        }

        private float m_roughnessAngleX;

        public float RoughnessAngleX
        {
            get
            {
                Debug.Assert(Type == MaterialType.Torrance || Type == MaterialType.Walter
                             || Type == MaterialType.Orennayar);
                return m_roughnessAngleX;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Torrance || Type == MaterialType.Walter
                             || Type == MaterialType.Orennayar);
                if (Equals(value, m_roughnessAngleX)) return;
                m_roughnessAngleX = value;
                OnPropertyChanged(nameof(RoughnessAngleX));
            }
        }

        private float m_roughnessAngleY;

        public float RoughnessAngleY
        {
            get
            {
                Debug.Assert(Type == MaterialType.Torrance || Type == MaterialType.Walter
                             || Type == MaterialType.Orennayar);
                return m_roughnessAngleY;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Torrance || Type == MaterialType.Walter
                             || Type == MaterialType.Orennayar);
                if (Equals(value, m_roughnessAngleY)) return;
                m_roughnessAngleY = value;
                OnPropertyChanged(nameof(RoughnessAngleY));
            }
        }

        private string m_roughnessTex = String.Empty;

        public string RoughnessTex
        {
            get
            {
                Debug.Assert(Type == MaterialType.Torrance || Type == MaterialType.Walter
                             || Type == MaterialType.Orennayar);
                return m_roughnessTex;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Torrance || Type == MaterialType.Walter
                             || Type == MaterialType.Orennayar);
                if (Equals(value, m_roughnessTex)) return;
                m_roughnessTex = value;
                OnPropertyChanged(nameof(RoughnessTex));
            }
        }

        private Vec3<float> m_absorption;

        public Vec3<float> Absorption
        {
            get
            {
                Debug.Assert(Type == MaterialType.Walter);
                return m_absorption;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Walter);
                if (Equals(value, m_absorption)) return;
                m_absorption = value;
                OnPropertyChanged(nameof(Absorption));
            }
        }

        private Vec3<float> m_radiance;

        public Vec3<float> Radiance
        {
            get
            {
                Debug.Assert(Type == MaterialType.Emmisive);
                return m_radiance;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Emmisive);
                if (Equals(m_radiance, value)) return;
                m_radiance = value;
                OnPropertyChanged(nameof(Radiance));
            }
        }

        private string m_radianceTex = String.Empty;

        public string RadianceTex
        {
            get
            {
                Debug.Assert(Type == MaterialType.Emmisive);
                return m_radianceTex;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Emmisive);
                if (Equals(m_radianceTex, value)) return;
                m_radianceTex = value;
                OnPropertyChanged(nameof(RadianceTex));
            }
        }

        private float m_scale = 1.0f;

        public float Scale
        {
            get
            {
                Debug.Assert(Type == MaterialType.Emmisive);
                return m_scale;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Emmisive);
                if (Equals(value, m_scale)) return;
                m_scale = value;
                OnPropertyChanged(nameof(Scale));
            }
        }

        private MaterialModel m_layerA = null;

        public MaterialModel LayerA
        {
            get
            {
                Debug.Assert(Type == MaterialType.Blend);
                return m_layerA;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Blend);
                if (ReferenceEquals(value, m_layerA)) return;
                m_layerA = value;
                OnPropertyChanged(nameof(LayerA));
            }
        }

        private MaterialModel m_layerB = null;

        public MaterialModel LayerB
        {
            get
            {
                Debug.Assert(Type == MaterialType.Blend);
                return m_layerB;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Blend);
                if (ReferenceEquals(value, m_layerB)) return;
                m_layerB = value;
                OnPropertyChanged(nameof(LayerB));
            }
        }

        private float m_factorA;

        public float FactorA
        {
            get
            {
                Debug.Assert(Type == MaterialType.Blend);
                return m_factorA;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Blend);
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_factorA) return;
                m_factorA = value;
                OnPropertyChanged(nameof(FactorA));
            }
        }

        private float m_factorB;

        public float FactorB
        {
            get
            {
                Debug.Assert(Type == MaterialType.Blend);
                return m_factorB;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Blend);
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_factorB) return;
                m_factorB = value;
                OnPropertyChanged(nameof(FactorB));
            }
        }

        private float m_refractionIndex;

        public float RefractionIndex
        {
            get
            {
                Debug.Assert(Type == MaterialType.Fresnel);
                return m_refractionIndex;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Fresnel);
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_refractionIndex) return;
                m_refractionIndex = value;
                OnPropertyChanged(nameof(RefractionIndex));
            }
        }

        private Vec2<float> m_refractionComplex;

        public Vec2<float> RefractionComplex
        {
            get
            {
                Debug.Assert(Type == MaterialType.Fresnel);
                return m_refractionComplex;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Fresnel);
                if (Equals(value, m_refractionComplex)) return;
                m_refractionComplex = value;
                OnPropertyChanged(nameof(RefractionComplex));
            }
        }

        private MaterialModel m_layerReflection = null;

        public MaterialModel LayerReflection
        {
            get
            {
                Debug.Assert(Type == MaterialType.Fresnel);
                return m_layerReflection;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Fresnel);
                if (ReferenceEquals(value, m_layerReflection)) return;
                m_layerReflection = value;
                OnPropertyChanged(nameof(LayerReflection));
            }
        }

        private MaterialModel m_layerRefraction = null;

        public MaterialModel LayerRefraction
        {
            get
            {
                Debug.Assert(Type == MaterialType.Fresnel);
                return m_layerRefraction;
            }
            set
            {
                Debug.Assert(Type == MaterialType.Fresnel);
                if (ReferenceEquals(value, m_layerRefraction)) return;
                m_layerRefraction = value;
                OnPropertyChanged(nameof(LayerRefraction));
            }
        }

        #region PropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
}
