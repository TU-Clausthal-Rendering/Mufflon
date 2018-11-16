using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.ViewModel.Light;

namespace gui.Model.Light
{
    public class EnvmapLightModel : LightModel
    {
        public override LightType Type => LightType.Envmap;

        public override LightViewModel CreateViewModel(Models models)
        {
            return new EnvmapLightViewModel(models, this);
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
