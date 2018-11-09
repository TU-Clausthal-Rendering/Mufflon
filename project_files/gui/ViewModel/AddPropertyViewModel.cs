using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using gui.Annotations;
using gui.Model.Light;
using gui.View.Helper;

namespace gui.ViewModel
{
    /// <summary>
    /// view model for the AddPropertyDialog.
    /// NameValue - final name for the property
    /// TypeValue - final type for the property
    /// </summary>
    public abstract class AddPropertyViewModel<T> : INotifyPropertyChanged
    {
        public abstract string WindowTitle { get; }

        public abstract string NameName { get; }

        public Visibility NameVisibility { get; }

        private string m_name = "";

        // public getter for the value (+ setter for the view)
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

        public abstract string TypeName { get; }

        public ReadOnlyObservableCollection<ComboBoxItem<T>> TypeList { get; }

        private ComboBoxItem<T> m_selectedType;

        public ComboBoxItem<T> SelectedType
        {
            get => m_selectedType;
            set
            {
                if (value == null || ReferenceEquals(value, m_selectedType)) return;
                m_selectedType = value;
                OnPropertyChanged(nameof(SelectedType));
            }
        }

        // public getter for the type
        public T TypeValue => m_selectedType.Cargo;

        /// <param name="types">items for the combo box</param>
        /// <param name="isNameVisible">indicates if the name property should be shown</param>
        protected AddPropertyViewModel(ReadOnlyObservableCollection<ComboBoxItem<T>> types, bool isNameVisible = true)
        {
            Debug.Assert(types.Count > 0);
            TypeList = types;
            NameVisibility = isNameVisible ? Visibility.Visible : Visibility.Hidden;
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
