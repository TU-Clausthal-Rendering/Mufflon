using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model.Material;
using gui.View.Helper;

namespace gui.ViewModel.Material
{
    /// <inheritdoc />
    public class AddMaterialViewModel : AddPropertyViewModel<MaterialModel.MaterialType>
    {
        public override string WindowTitle => "Add Material";
        public override string NameName => "Material Name:";
        public override string TypeName => "Type:";

        /// <param name="isNameVisible">indicates if the name property should be shown</param>
        public AddMaterialViewModel(bool isNameVisible) : base(
            new ReadOnlyObservableCollection<ComboBoxItem<MaterialModel.MaterialType>>(
            new ObservableCollection<ComboBoxItem<MaterialModel.MaterialType>>
            {
                new ComboBoxItem<MaterialModel.MaterialType>(MaterialModel.MaterialType.Lambert),           
                new ComboBoxItem<MaterialModel.MaterialType>(MaterialModel.MaterialType.Torrance),           
                new ComboBoxItem<MaterialModel.MaterialType>(MaterialModel.MaterialType.Walter),           
                new ComboBoxItem<MaterialModel.MaterialType>(MaterialModel.MaterialType.Emissive),           
                new ComboBoxItem<MaterialModel.MaterialType>(MaterialModel.MaterialType.Orennayar),           
                new ComboBoxItem<MaterialModel.MaterialType>(MaterialModel.MaterialType.Blend),           
                new ComboBoxItem<MaterialModel.MaterialType>(MaterialModel.MaterialType.Fresnel)           
            }), isNameVisible)
        {}
    }
}
