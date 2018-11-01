using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using gui.Annotations;
using gui.Model.Camera;

namespace gui.ViewModel.Camera
{
    public abstract class CameraViewModel : INotifyPropertyChanged
    {
        private readonly CameraModel m_parent;

        protected CameraViewModel(CameraModel parent)
        {
            m_parent = parent;
            parent.PropertyChanged += ModelOnPropertyChanged;
        }

        protected virtual void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(CameraModel.Name):
                    OnPropertyChanged(nameof(Name));
                    break;
                case nameof(CameraModel.Type):
                    OnPropertyChanged(nameof(Type));
                    break;
                case nameof(CameraModel.IsSelected):
                    OnPropertyChanged(nameof(IsSelected));
                    break;
            }
        }

        public string Type => m_parent.Type.ToString();

        public string Name
        {
            get => m_parent.Name;
            set => m_parent.Name = value;
        }

        public bool IsSelected
        {
            get => m_parent.IsSelected;
            set => m_parent.IsSelected = value;
        }

        /// <summary>
        /// create a new view based on this view model
        /// </summary>
        /// <returns></returns>
        public abstract object CreateView();

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
