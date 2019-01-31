using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using gui.Annotations;
using gui.Command;
using gui.Model;
using gui.Model.Camera;
using gui.Model.Scene;

namespace gui.ViewModel.Camera
{
    public abstract class CameraViewModel : INotifyPropertyChanged
    {
        private readonly WorldModel m_world;
        private readonly CameraModel m_parent;

        protected CameraViewModel(Models models, CameraModel parent)
        {
            m_world = models.World;
            m_parent = parent;
            parent.PropertyChanged += ModelOnPropertyChanged;

            m_world.CurrentScenario.PropertyChanged += CurrentScenarioOnPropertyChanged;
            m_world.PropertyChanged += WorldOnPropertyChanged;
        }

        private void WorldOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(WorldModel.CurrentScenario):
                    m_world.PreviousScenario.PropertyChanged -= CurrentScenarioOnPropertyChanged;
                    m_world.CurrentScenario.PropertyChanged += CurrentScenarioOnPropertyChanged;
                    OnPropertyChanged(nameof(IsSelected));
                    break;
            }
        }

        private void CurrentScenarioOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(ScenarioModel.Camera):
                    OnPropertyChanged(nameof(IsSelected));
                    break;
            }
        }

        protected virtual void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(CameraModel.Name):
                    OnPropertyChanged(nameof(Name));
                    break;
                case nameof(CameraModel.Type):
                    OnPropertyChanged(nameof(Type));
                    break;
            }
        }

        public string Type => m_parent.Type.ToString();

        public string Name => m_parent.Name;

        public bool IsSelected
        {
            get => ReferenceEquals(m_world.CurrentScenario.Camera, m_parent);
            set
            {
                if(value)
                    m_world.CurrentScenario.Camera = m_parent;
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
