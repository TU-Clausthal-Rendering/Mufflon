using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using gui.Annotations;
using gui.Model.Light;

namespace gui.ViewModel.Light
{
    public abstract class LightViewModel : INotifyPropertyChanged
    {
        private readonly LightModel m_parent;

        protected LightViewModel(LightModel parent)
        {
            m_parent = parent;
            parent.PropertyChanged += ModelOnPropertyChanged;
        }

        protected virtual void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(LightModel.Name):
                    OnPropertyChanged(nameof(Name));
                    break;
                case nameof(LightModel.Scale):
                    OnPropertyChanged(nameof(Scale));
                    break;
                case nameof(LightModel.IsSelected):
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

        public float Scale
        {
            get => m_parent.Scale;
            set => m_parent.Scale = value;
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
