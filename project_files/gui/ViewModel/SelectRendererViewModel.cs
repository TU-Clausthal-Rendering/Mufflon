using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Dll;
using gui.Model.Camera;
using gui.View.Helper;

namespace gui.ViewModel
{
    /// <inheritdoc />
    public class SelectRendererViewModel : AddPropertyViewModel<Core.RendererType>
    {
        // We're 'abusing' the add-property base viewmodel (we don't need some of the labels,
        // so they're just being ignored)
        public override string WindowTitle => "Select renderer";
        public override string NameName => "";
        public override string TypeName => "";

        private static ObservableCollection<ComboBoxItem<Core.RendererType>> createCombobox()
        {
            var box = new ObservableCollection<ComboBoxItem<Core.RendererType>>();
            
            foreach(Core.RendererType renderer in Enum.GetValues(typeof(Core.RendererType)))
            {
                box.Add(new ComboBoxItem<Core.RendererType>(renderer));
            }

            return box;
        }

        public SelectRendererViewModel() : base(
            new ReadOnlyObservableCollection<ComboBoxItem<Core.RendererType>>(createCombobox()))
        { }
    }
}
