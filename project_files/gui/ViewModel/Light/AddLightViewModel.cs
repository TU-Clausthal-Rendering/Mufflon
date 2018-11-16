using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using gui.Annotations;
using gui.Model.Light;
using gui.View.Helper;

namespace gui.ViewModel.Light
{
    /// <inheritdoc />
    public class AddLightViewModel : AddPropertyViewModel<LightModel.LightType>
    {
        public override string WindowTitle => "Add Light";
        public override string NameName => "Light Name:";
        public override string TypeName => "Type:";

        public AddLightViewModel() : base( 
            new ReadOnlyObservableCollection<ComboBoxItem<LightModel.LightType>>(
            new ObservableCollection<ComboBoxItem<LightModel.LightType>>
            {
                new ComboBoxItem<LightModel.LightType>(LightModel.LightType.Point),
                new ComboBoxItem<LightModel.LightType>(LightModel.LightType.Directional),
                new ComboBoxItem<LightModel.LightType>(LightModel.LightType.Spot),
                new ComboBoxItem<LightModel.LightType>(LightModel.LightType.Envmap),
                new ComboBoxItem<LightModel.LightType>(LightModel.LightType.Goniometric)
            }))
        {}
    }
}
