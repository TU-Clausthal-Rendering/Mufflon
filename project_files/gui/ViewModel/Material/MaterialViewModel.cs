using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using gui.Annotations;
using gui.Command;
using gui.Model;
using gui.Model.Material;
using gui.View.Material;

namespace gui.ViewModel.Material
{
    public abstract class MaterialViewModel : INotifyPropertyChanged
    {
        private readonly MaterialModel m_parent;

        protected MaterialViewModel(Models models, MaterialModel parent)
        {
            this.m_parent = parent;
            parent.PropertyChanged += ModelOnPropertyChanged;
        }

        protected virtual void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            // add name change if it will ever be available
        }

        public UIElement CreateView()
        {
            if (m_parent.IsRecursive)
                return new RecursiveMaterialView(this, CreateInternalView());
            return new MaterialView(this, CreateInternalView());
        }

        protected abstract UIElement CreateInternalView();

        public string Name => m_parent.Name;

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
