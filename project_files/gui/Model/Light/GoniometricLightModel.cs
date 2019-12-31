using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model.Scene;
using gui.Utility;
using gui.ViewModel.Light;

namespace gui.Model.Light
{
    public class GoniometricLightModel : LightModel
    {
        public override LightType Type => LightType.Goniometric;

        public override LightViewModel CreateViewModel(Models models)
        {
            return new GoniometricLightViewModel(models, this);
        }

        public GoniometricLightModel(UInt32 handle) : base(handle)
        {

        }

        public override float Scale
        {
            get => 1f;
            set
            {
                if (value == Scale) return;
                OnPropertyChanged(nameof(Scale));
            }
        }

        private Vec3<float> m_position;

        public Vec3<float> Position
        {
            get => m_position;
            set
            {
                if (Equals(m_position, value)) return;
                m_position = value;
                OnPropertyChanged(nameof(Position));
            }
        }
        
        private string m_map = String.Empty;

        public string Map
        {
            get => m_map;
            set
            {
                if (Equals(value, m_map)) return;
                m_map = value;
                OnPropertyChanged(nameof(Map));
            }
        }
    }
}
