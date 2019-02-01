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
    public class SelectRendererViewModel : AddPropertyViewModel<UInt32>
    {
        // We're 'abusing' the add-property base viewmodel (we don't need some of the labels,
        // so they're just being ignored)
        public override string WindowTitle => "Select renderer";
        public override string NameName => "";
        public override string TypeName => "";

        private static ObservableCollection<string> m_rendererNames = new ObservableCollection<string>();

        private static ObservableCollection<ComboBoxItem<UInt32>> createCombobox()
        {
            var box = new ObservableCollection<ComboBoxItem<UInt32>>();
            for(UInt32 i = 0u; i < Core.render_get_renderer_count(); ++i)
            {
                box.Add(new ComboBoxItem<UInt32>(i));
                m_rendererNames.Add(Core.render_get_renderer_name(i));
            }
            
            return box;
        }

        public SelectRendererViewModel() : base(
            new ReadOnlyObservableCollection<ComboBoxItem<UInt32>>(createCombobox()))
        { }
    }
}
