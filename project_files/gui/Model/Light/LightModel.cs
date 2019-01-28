using System;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using gui.Annotations;
using gui.Dll;
using gui.Model.Scene;
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

        protected LightModel(IntPtr handle)
        {
            Handle = handle;
        }

        public abstract LightType Type { get; }

        public string Name => Core.world_get_light_name(Handle);

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

        public IntPtr Handle { get; }

        /// <summary>
        /// creates a new view model based on this model
        /// </summary>
        /// <param name="models"></param>
        /// <returns></returns>
        public abstract LightViewModel CreateViewModel(Models models);

        public static LightModel MakeFromHandle(IntPtr handle, LightType type)
        {
            switch (type)
            {
                case LightType.Point:
                    return new PointLightModel(handle);
                case LightType.Directional:
                    return new DirectionalLightModel(handle);
                case LightType.Spot:
                    return new SpotLightModel(handle);
                case LightType.Envmap:
                    return new EnvmapLightModel(handle);
                case LightType.Goniometric:
                default:
                    // not implemented
                    Debug.Assert(false);
                    throw new NotImplementedException();
            }
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
