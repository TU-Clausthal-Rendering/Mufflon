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

        protected MaterialModel GetModel(MaterialModel.MaterialType type, bool isRecursive, Action<MaterialModel> removeAction)
        {
            switch (type)
            {
                case MaterialModel.MaterialType.Lambert:
                    return new LambertMaterialModel(isRecursive, removeAction);
                case MaterialModel.MaterialType.Torrance:
                    return new TorranceMaterialModel(isRecursive, removeAction);
                case MaterialModel.MaterialType.Walter:
                    return new WalterMaterialModel(isRecursive, removeAction);
                case MaterialModel.MaterialType.Emissive:
                    return new EmissiveMaterialModel(isRecursive, removeAction);
                case MaterialModel.MaterialType.Orennayar:
                    return new OrennayarMaterialModel(isRecursive, removeAction);
                case MaterialModel.MaterialType.Blend:
                    return new BlendMaterialModel(isRecursive, removeAction);
                case MaterialModel.MaterialType.Fresnel:
                    return new FresnelMaterialModel(isRecursive, removeAction);
            }

            return null;
        }

        public virtual void Execute(object parameter)
        {
            var dc = new AddMaterialViewModel(true);
            var dialog = new AddPropertyDialog(dc);

            if (dialog.ShowDialog() != true) return;

            MaterialModel mm = GetModel(dc.TypeValue, false, model => m_models.Materials.Models.Remove(model));
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
