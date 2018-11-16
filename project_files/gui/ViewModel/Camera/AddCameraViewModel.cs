using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model.Camera;
using gui.View.Helper;

namespace gui.ViewModel.Camera
{
    /// <inheritdoc />
    public class AddCameraViewModel : AddPropertyViewModel<CameraModel.CameraType>
    {
        public override string WindowTitle => "Add Camera";
        public override string NameName => "Camera Name:";
        public override string TypeName => "Type:";

        public AddCameraViewModel() : base(
            new ReadOnlyObservableCollection<ComboBoxItem<CameraModel.CameraType>>(
            new ObservableCollection<ComboBoxItem<CameraModel.CameraType>>
            {
                new ComboBoxItem<CameraModel.CameraType>(CameraModel.CameraType.Pinhole),
                new ComboBoxItem<CameraModel.CameraType>(CameraModel.CameraType.Focus),
                new ComboBoxItem<CameraModel.CameraType>(CameraModel.CameraType.Ortho),
            }))
        {}
    }
}
