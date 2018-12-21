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

        /// <summary>
        /// required for torrance and walter roughness
        /// </summary>
        public enum RoughnessType
        {
            Isotropic,
            Anisotropic,
            Texture
        }

        /// <param name="isRecursive">indicates if this material is included in another material (i.e. blend or fresnel)</param>
        /// <param name="removeAction">
        /// action that removes the material (either from the model list or from another material model if it was recursive)
        /// The MaterialModel parameter will be the this pointer
        /// </param>
        protected MaterialModel(bool isRecursive, Action<MaterialModel> removeAction)
        {
            IsRecursive = isRecursive;
            m_removeAction = removeAction;
        }

        public void Remove()
        {
            m_removeAction.Invoke(this);
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

        private IntPtr m_handle = IntPtr.Zero;
        public IntPtr Handle
        {
            get => m_handle;
            set
            {
                if (m_handle == value) return;
                m_handle = value;
                OnPropertyChanged(nameof(Handle));
            }
        }

        // indicates if this material is included in another material (i.e. blend or fresnel)
        public bool IsRecursive { get; }

        private readonly Action<MaterialModel> m_removeAction;

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
