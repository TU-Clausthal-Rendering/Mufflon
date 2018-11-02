using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using gui.Annotations;
using gui.Utility;
using gui.ViewModel.Material;

namespace gui.Model.Material
{
    public abstract class MaterialModel : INotifyPropertyChanged
    {
        public enum MaterialType
        {
            Lambert,
            Torrance,
            Walter,
            Emissive,
            Orennayar,
            Blend,
            Fresnel
        }
        
        public abstract MaterialType Type { get; }

        private string m_name = String.Empty;

        public string Name
        {
            get => m_name;
            set
            {
                if (value == null || value == m_name) return;
                m_name = value;
                OnPropertyChanged(nameof(Name));
            }
        }

        /// <summary>
        /// creates a new view model based on this model
        /// </summary>
        /// <param name="models"></param>
        /// <returns></returns>
        public abstract MaterialViewModel CreateViewModel(Models models);

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
