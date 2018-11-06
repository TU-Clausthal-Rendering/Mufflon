using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Model;
using gui.Model.Material;
using gui.View.Dialog;
using gui.ViewModel.Material;

namespace gui.Command
{
    public class AddMaterialCommand : ICommand
    {
        private readonly Models m_models;

        public AddMaterialCommand(Models models)
        {
            m_models = models;
        }

        public bool CanExecute(object parameter)
        {
            return true;
        }

        public void Execute(object parameter)
        {
            var dc = new AddMaterialViewModel();
            var dialog = new AddPropertyDialog(dc);

            if (dialog.ShowDialog() != true) return;

            MaterialModel mm = null;
            switch (dc.TypeValue)
            {
                case MaterialModel.MaterialType.Lambert:
                    mm = new LambertMaterialModel();
                    break;
                case MaterialModel.MaterialType.Torrance:
                    mm = new TorranceMaterialModel();
                    break;
                case MaterialModel.MaterialType.Walter:
                    mm = new WalterMaterialModel();
                    break;
                case MaterialModel.MaterialType.Emissive:
                    mm = new EmissiveMaterialModel();
                    break;
                case MaterialModel.MaterialType.Orennayar:
                    mm = new OrennayarMaterialModel();
                    break;
                case MaterialModel.MaterialType.Blend:
                    mm = new BlendMaterialModel();
                    break;
                case MaterialModel.MaterialType.Fresnel:
                    mm = new FresnelMaterialModel();
                    break;
            }
            Debug.Assert(mm != null);

            mm.Name = dc.NameValue;

            m_models.Materials.Models.Add(mm);
        }

        public event EventHandler CanExecuteChanged
        {
            add { }
            remove { }
        }
    }
}
