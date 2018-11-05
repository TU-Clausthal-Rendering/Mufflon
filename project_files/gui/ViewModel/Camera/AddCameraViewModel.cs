using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using gui.Annotations;
using gui.Model.Light;
using gui.View.Helper;

namespace gui.ViewModel.Camera
{
    public class AddCameraViewModel : INotifyPropertyChanged
    {
        public string WindowTitle => "Add Material";

        public string NameName => "Material:";

        private string m_name = "";

        public string NameValue
        {
            get => m_name;
            set
            {
                if (value == null || value == m_name) return;
                m_name = value;
                OnPropertyChanged(nameof(NameValue));
            }
        }

        public string TypeName => "Type:";

        public ReadOnlyObservableCollection<ComboBoxItem<LightModel.LightType>> TypeList { get; }

        private ComboBoxItem<LightModel.LightType> m_selectedType;

        public ComboBoxItem<LightModel.LightType> SelectedType
        {
            get => m_selectedType;
            set
            {
                if (value == null || ReferenceEquals(value, m_selectedType)) return;
                m_selectedType = value;
                OnPropertyChanged(nameof(SelectedType));
            }
        }

        public AddCameraViewModel()
        {
            var types = new ObservableCollection<ComboBoxItem<LightModel.LightType>>
            {
                new ComboBoxItem<LightModel.LightType>(LightModel.LightType.Point),
                new ComboBoxItem<LightModel.LightType>(LightModel.LightType.Directional),
                new ComboBoxItem<LightModel.LightType>(LightModel.LightType.Spot),
                new ComboBoxItem<LightModel.LightType>(LightModel.LightType.Envmap),
                new ComboBoxItem<LightModel.LightType>(LightModel.LightType.Goniometric)
            };
            TypeList = new ReadOnlyObservableCollection<ComboBoxItem<LightModel.LightType>>(types);

            m_selectedType = types.First();
        }

        #region PropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion  
    }
}
