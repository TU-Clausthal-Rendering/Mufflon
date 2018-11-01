using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model.Light;

namespace gui.ViewModel.Light
{
    public class EnvmapLightViewModel : LightViewModel
    {
        private readonly EnvmapLightModel m_parent;

        public EnvmapLightViewModel(EnvmapLightModel parent) : base(parent)
        {
            m_parent = parent;
        }

        protected override void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            base.ModelOnPropertyChanged(sender, args);
            switch (args.PropertyName)
            {
                case nameof(EnvmapLightModel.Map):
                    OnPropertyChanged(nameof(MapFull));
                    OnPropertyChanged(nameof(MapShort));
                    break;
            }
        }

        public override object CreateView()
        {
            throw new NotImplementedException();
        }

        // readonly property
        public string MapFull => m_parent.Map;

        public string MapShort => Path.GetFileName(m_parent.Map);
    }
}
