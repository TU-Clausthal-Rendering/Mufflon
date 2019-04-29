using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Annotations;
using gui.Command;
using gui.Model;
using gui.Model.Light;
using gui.Model.Scene;

namespace gui.ViewModel.Light
{
    public abstract class LightViewModel : INotifyPropertyChanged
    {
        private readonly WorldModel m_world;
        private readonly LightModel m_parent;

        protected LightViewModel(Models models, LightModel parent)
        {
            m_world = models.World;
            m_parent = parent;
            parent.PropertyChanged += ModelOnPropertyChanged;

            m_world.CurrentScenario.Lights.CollectionChanged += ScenarioLightsOnCollectionChanged;
            m_world.PropertyChanged += WorldOnPropertyChanged;
        }

        private void WorldOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(WorldModel.CurrentScenario):
                    m_world.PreviousScenario.Lights.CollectionChanged -= ScenarioLightsOnCollectionChanged;
                    m_world.CurrentScenario.Lights.CollectionChanged += ScenarioLightsOnCollectionChanged;
                    OnPropertyChanged(nameof(IsSelected));
                    break;
            }
        }

        private void ScenarioLightsOnCollectionChanged(object sender, NotifyCollectionChangedEventArgs e)
        {
            // is selected might have changed
            OnPropertyChanged(nameof(IsSelected));
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
            }
        }

        public string Type => m_parent.Type.ToString();

        public string Name => m_parent.Name;

        public float Scale
        {
            get => m_parent.Scale;
            set => m_parent.Scale = value;
        }

        public bool IsAnimated { get => m_parent.PathSegments > 1u; }

        public bool IsSelected
        {
            get => m_world.CurrentScenario.Lights.Contains(m_parent.Handle);
            set
            {
                if (value == IsSelected) return;
                if (value)
                    m_world.CurrentScenario.Lights.Add(m_parent.Handle);
                else
                    m_world.CurrentScenario.Lights.Remove(m_parent.Handle);
            }
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
