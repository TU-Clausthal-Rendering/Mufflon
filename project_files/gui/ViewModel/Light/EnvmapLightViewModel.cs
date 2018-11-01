using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using gui.Model;
using gui.Model.Light;

namespace gui.ViewModel.Light
{
    public class EnvmapLightViewModel : LightViewModel
    {
        private class SelectMapCommand : ICommand
        {
            private readonly EnvmapLightModel m_model;
            private readonly Models m_models;

            public SelectMapCommand(EnvmapLightModel model, Models models)
            {
                m_model = model;
                m_models = models;
            }
   
            public bool CanExecute(object parameter)
            {
                return true;
            }

            public void Execute(object parameter)
            {
                var ofd = new Microsoft.Win32.OpenFileDialog
                {
                    Multiselect = false,
                    InitialDirectory = Properties.Settings.Default.ImagePath
                };

                if (ofd.ShowDialog(m_models.App.Window) != true) return;

                Properties.Settings.Default.ImagePath = System.IO.Path.GetDirectoryName(ofd.FileName);
                // TODO file name relative?
                m_model.Map = ofd.FileName;
            }

            public event EventHandler CanExecuteChanged;
        }

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

        public string MapShort
        {
            get => Path.GetFileName(m_parent.Map);
            set { } // dummy setter because of textbox (requires setter but is disabled)
        }
    }
}
