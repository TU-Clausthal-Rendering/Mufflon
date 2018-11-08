using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model;
using gui.Model.Material;
using gui.View.Dialog;
using gui.ViewModel.Material;

namespace gui.Command
{
    public class AddRecursiveMaterialCommand : AddMaterialCommand
    {
        private readonly Action<MaterialModel> m_setter;

        public AddRecursiveMaterialCommand(Models models, Action<MaterialModel> setter) : base(models)
        {
            m_setter = setter;
        }

        public override void Execute(object parameter)
        {
            var dc = new AddMaterialViewModel();
            var dialog = new AddPropertyDialog(dc);

            if (dialog.ShowDialog() != true) return;

            MaterialModel mm = GetModel(dc.TypeValue);
            Debug.Assert(mm != null);

            m_setter.Invoke(mm);
        }
    }
}
