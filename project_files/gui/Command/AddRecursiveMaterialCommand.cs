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
        private readonly Action<MaterialModel> m_remover;
        private readonly string m_title;

        public AddRecursiveMaterialCommand(Models models, Action<MaterialModel> setter, Action<MaterialModel> remover, string title) : base(models)
        {
            m_setter = setter;
            m_remover = remover;
            m_title = title;
        }

        public override void Execute(object parameter)
        {
            var dc = new AddMaterialViewModel(false);
            var dialog = new AddPropertyDialog(dc);

            if (dialog.ShowDialog() != true) return;

            MaterialModel mm = GetModel(dc.TypeValue, true, m_remover);
            Debug.Assert(mm != null);
            mm.Name = m_title;

            m_setter.Invoke(mm);
        }
    }
}
