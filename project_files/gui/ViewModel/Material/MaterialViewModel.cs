using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using gui.Annotations;
using gui.Model.Material;

namespace gui.ViewModel.Material
{
    public abstract class MaterialViewModel : INotifyPropertyChanged
    {
        private readonly MaterialModel m_parent;

        protected MaterialViewModel(MaterialModel parent)
        {
            this.m_parent = parent;
            parent.PropertyChanged += ModelOnPropertyChanged;
        }

        protected virtual void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(MaterialModel.Name):
                    OnPropertyChanged(nameof(Name));
                    break;
            }
        }

        public abstract object CreateView();

        public string Name
        {
            get => m_parent.Name;
            set => m_parent.Name = value;
        }

        public string Type => m_parent.Type.ToString();

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
