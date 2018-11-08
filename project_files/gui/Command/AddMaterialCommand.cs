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

        protected MaterialModel GetModel(MaterialModel.MaterialType type)
        {
            switch (type)
            {
                case MaterialModel.MaterialType.Lambert:
                    return new LambertMaterialModel();
                case MaterialModel.MaterialType.Torrance:
                    return new TorranceMaterialModel();
                case MaterialModel.MaterialType.Walter:
                    return new WalterMaterialModel();
                case MaterialModel.MaterialType.Emissive:
                    return new EmissiveMaterialModel();
                case MaterialModel.MaterialType.Orennayar:
                    return new OrennayarMaterialModel();
                case MaterialModel.MaterialType.Blend:
                    return new BlendMaterialModel();
                case MaterialModel.MaterialType.Fresnel:
                    return new FresnelMaterialModel();
            }

            return null;
        }

        public virtual void Execute(object parameter)
        {
            var dc = new AddMaterialViewModel();
            var dialog = new AddPropertyDialog(dc);

            if (dialog.ShowDialog() != true) return;

            MaterialModel mm = GetModel(dc.TypeValue);
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
