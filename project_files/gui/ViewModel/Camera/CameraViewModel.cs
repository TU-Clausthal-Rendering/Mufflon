using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using gui.Annotations;
using gui.Command;
using gui.Model;
using gui.Model.Camera;
using gui.Model.Scene;
using gui.Dll;

namespace gui.ViewModel.Camera
{
    public abstract class CameraViewModel : INotifyPropertyChanged
    {
        private readonly WorldModel m_world;
        private readonly CameraModel m_parent;
        private readonly ResetCameraCommand m_resetTransRotCommand;

        protected CameraViewModel(Models models, CameraModel parent)
        {
            m_world = models.World;
            m_parent = parent;
            m_resetTransRotCommand = new ResetCameraCommand(models.Settings, parent);
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
                case nameof(CameraModel.Position):
                    OnPropertyChanged(nameof(PositionX));
                    OnPropertyChanged(nameof(PositionY));
                    OnPropertyChanged(nameof(PositionZ));
                    break;
                case nameof(CameraModel.ViewDirection):
                    OnPropertyChanged(nameof(DirectionX));
                    OnPropertyChanged(nameof(DirectionY));
                    OnPropertyChanged(nameof(DirectionZ));
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

        public float PositionX => m_parent.Position.X;
        public float PositionY => m_parent.Position.Y;
        public float PositionZ => m_parent.Position.Z;
        public float DirectionX => m_parent.ViewDirection.X;
        public float DirectionY => m_parent.ViewDirection.Y;
        public float DirectionZ => m_parent.ViewDirection.Z;

        // TODO: couple this with a display of position/view direction
        public ResetCameraCommand ResetTransRotCommand => m_resetTransRotCommand;

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
