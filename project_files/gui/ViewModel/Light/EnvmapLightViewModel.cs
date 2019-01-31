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
using gui.Model;
using gui.Model.Light;
using gui.View.Light;

namespace gui.ViewModel.Light
{
    public class EnvmapLightViewModel : LightViewModel
    {
        private readonly EnvmapLightModel m_parent;

        public EnvmapLightViewModel(Models world, EnvmapLightModel parent) : base(world, parent)
        {
            m_parent = parent;
            SelectMapCommand = new SelectTextureCommand(world, () => m_parent.Map, val => m_parent.Map = val);
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

        // readonly property
        public string Map
        {
            get => m_parent.Map;
            set => m_parent.Map = value;
        }

        public ICommand SelectMapCommand { get; }
    }
}
