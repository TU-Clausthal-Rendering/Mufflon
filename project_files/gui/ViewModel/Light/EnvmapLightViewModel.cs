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

        public string Map
        {
            get => m_parent.Map;
            set
            {
                var absolutePath = Path.Combine(m_world.Directory, value);
                m_parent.Map = absolutePath;
            }
        }

        public ICommand SelectMapCommand { get; }
    }
}
