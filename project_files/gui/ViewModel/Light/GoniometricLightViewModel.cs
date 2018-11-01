using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model.Light;

namespace gui.ViewModel.Light
{
    public class GoniometricLightViewModel : LightViewModel
    {
        private readonly GoniometricLightModel m_parent;

        public GoniometricLightViewModel(GoniometricLightModel parent) : base(parent)
        {
            m_parent = parent;
        }

        public override object CreateView()
        {
            throw new NotImplementedException();
        }
    }
}
