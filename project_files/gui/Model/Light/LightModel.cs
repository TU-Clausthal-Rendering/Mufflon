using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using gui.Annotations;
using gui.Utility;
using gui.ViewModel.Light;

namespace gui.Model.Light
{
    public abstract class LightModel : INotifyPropertyChanged
    {
        public enum LightType
        {
            Point,
            Directional,
            Spot,
            Envmap,
            Goniometric
        }

        public abstract LightType Type { get; }

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

        private float m_scale = 1.0f;

        public float Scale
        {
            get => m_scale;
            set
            {
                Debug.Assert(Scale >= 0.0f);
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                if (value == m_scale) return;
                m_scale = value;
                OnPropertyChanged(nameof(Scale));
            }
        }

        private bool m_isSelected = false;
        // indicates if this light should be used for the renderer
        public bool IsSelected
        {
            get => m_isSelected;
            set
            {
                if (value == m_isSelected) return;
                m_isSelected = value;
                OnPropertyChanged(nameof(IsSelected));
            }
        }

        /// <summary>
        /// creates a new view model based on this model
        /// </summary>
        /// <param name="models"></param>
        /// <returns></returns>
        public abstract LightViewModel CreateViewModel(Models models);

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
