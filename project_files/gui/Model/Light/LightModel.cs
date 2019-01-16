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

        protected readonly WorldModel m_world;
        private ScenarioModel m_scenario = null;

        protected LightModel(IntPtr handle, WorldModel world)
        {
            Handle = handle;
            m_world = world;
            m_scenario = world.CurrentScenario;
            m_world.PropertyChanged += WorldOnPropertyChanged;
        }

        /// <summary>
        /// tests if IsSelected was changed => light added or removed from current scenario
        /// </summary>
        private void CurrentScenarioLightsOnCollectionChanged(object sender, NotifyCollectionChangedEventArgs args)
        {
            var changed = 
                (args.NewItems != null && args.NewItems.Contains(Name))
                || (args.OldItems != null && args.OldItems.Contains(Name));

            if(changed)
                // TODO this should trigger reload scene
                OnPropertyChanged(nameof(IsSelected));
        }

        private void WorldOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(WorldModel.CurrentScenario):
                    // scenario is only null when the light is created (because the scenarios are not initialized yet)
                    if (m_scenario != null)
                    {
                        OnPropertyChanged(nameof(IsSelected));
                        // unsubscribe from old scenario and subscribe to new one
                        m_scenario.Lights.CollectionChanged -= CurrentScenarioLightsOnCollectionChanged;
                    }
                    m_scenario = m_world.CurrentScenario;
                    m_scenario.Lights.CollectionChanged += CurrentScenarioLightsOnCollectionChanged;
                    break;
            }
        }

        public abstract LightType Type { get; }

        public string Name
        {
            get => Core.world_get_light_name(Handle);
            set
            {
                if (value == null || value == Name) return;
                // TODO set core name
                Debug.Assert(false);
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

        // indicates if this light should be used for the current scenario
        public bool IsSelected
        {
            get => m_world.CurrentScenario.Lights.Contains(Name);
            set
            {
                if (value == IsSelected) return;
                if (value)
                    m_world.CurrentScenario.Lights.Add(Name);
                else
                    m_world.CurrentScenario.Lights.Remove(Name);
                // property changed will be raised by the collection
            }
        }

        public IntPtr Handle { get; }

        /// <summary>
        /// creates a new view model based on this model
        /// </summary>
        /// <param name="models"></param>
        /// <returns></returns>
        public abstract LightViewModel CreateViewModel(Models models);

        public static LightModel MakeFromHandle(IntPtr handle, LightType type, WorldModel world)
        {
            Debug.Assert(world != null);
            switch (type)
            {
                case LightType.Point:
                    return new PointLightModel(handle, world);
                case LightType.Directional:
                    return new DirectionalLightModel(handle, world);
                case LightType.Spot:
                    return new SpotLightModel(handle, world);
                case LightType.Envmap:
                    return new EnvmapLightModel(handle, world);
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
