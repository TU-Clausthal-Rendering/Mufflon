using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using gui.Command;
using gui.Dll;
using gui.Model;
using gui.Model.Light;
using gui.Model.Scene;
using gui.View.Light;

namespace gui.ViewModel.Light
{
    public class EnvmapLightViewModel : LightViewModel
    {
        private readonly WorldModel m_world;
        private readonly EnvmapLightModel m_parent;

        public EnvmapLightViewModel(Models models, EnvmapLightModel parent) : base(models, parent)
        {
            m_world = models.World;
            m_parent = parent;
            SelectMapCommand = new SelectTextureCommand(models, () => Map, val => Map = val);
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(EnvmapLightModel.Map):
                    OnPropertyChanged(nameof(Map));
                    break;
            }
        }

        public override object CreateView()
        {
            return new LightView(this, new EnvmapLightView());
        }

        public bool IsEnvmap
        {
            get => m_parent.EnvType == Core.BackgroundType.Envmap;
        }
        public bool IsMonochrome
        {
            get => m_parent.EnvType == Core.BackgroundType.Monochrome;
        }
        public bool IsSky
        {
            get => m_parent.EnvType == Core.BackgroundType.SkyHosek;
        }

        public string Map
        {
            get => m_parent.Map;
            set
            {
                var absolutePath = Path.Combine(m_world.Directory, value);
                m_parent.Map = absolutePath;
            }
        }

        public float Albedo
        {
            get => m_parent.Albedo;
            set => m_parent.Albedo = value;
        }

        public float SolarRadius
        {
            get => m_parent.SolarRadius;
            set => m_parent.SolarRadius = value;
        }

        public float Turbidity
        {
            get => m_parent.Turbidity;
            set => m_parent.Turbidity = value;
        }

        public float SunDirX
        {
            get => m_parent.SunDir.X;
            set => m_parent.SunDir = new Utility.Vec3<float>(value, m_parent.SunDir.Y, m_parent.SunDir.Z);
        }
        public float SunDirY
        {
            get => m_parent.SunDir.Y;
            set => m_parent.SunDir = new Utility.Vec3<float>(m_parent.SunDir.X, value, m_parent.SunDir.Z);
        }
        public float SunDirZ
        {
            get => m_parent.SunDir.Z;
            set => m_parent.SunDir = new Utility.Vec3<float>(m_parent.SunDir.X, m_parent.SunDir.Y, value);
        }

        public float ColorX
        {
            get => m_parent.Color.X;
            set => m_parent.Color = new Utility.Vec3<float>(value, m_parent.Color.Y, m_parent.Color.Z);
        }
        public float ColorY
        {
            get => m_parent.Color.Y;
            set => m_parent.Color = new Utility.Vec3<float>(m_parent.Color.X, value, m_parent.Color.Z);
        }
        public float ColorZ
        {
            get => m_parent.Color.Z;
            set => m_parent.Color = new Utility.Vec3<float>(m_parent.Color.X, m_parent.Color.Y, value);
        }

        public ICommand SelectMapCommand { get; }
    }
}
