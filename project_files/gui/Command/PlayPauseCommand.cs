using System;
using System.ComponentModel;
using System.Threading;
using System.Windows.Input;
using gui.Model;
using gui.Model.Scene;

namespace gui.Command
{
    public class PlayPauseCommand : ICommand
    {
        private readonly Models m_models;

        public PlayPauseCommand(Models models)
        {
            m_models = models;
            m_models.PropertyChanged += ModelsOnPropertyChanged;
            if (m_models.World != null)
                m_models.World.PropertyChanged += WorldOnPropertyChanged;
        }

        private void ModelsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(WorldModel):
                    if(m_models.World != null)
                        m_models.World.PropertyChanged += WorldOnPropertyChanged;
                    OnCanExecuteChanged();
                    break;
            }
        }

        private void WorldOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(WorldModel.IsSane):
                    OnCanExecuteChanged();
                    break;
            }
        }

        public bool CanExecute(object parameter)
        {
            return m_models.World != null && m_models.World.IsSane;
        }

        public void Execute(object parameter)
        {
            m_models.Renderer.IsRendering = !m_models.Renderer.IsRendering;
        }

        public event EventHandler CanExecuteChanged;

        protected virtual void OnCanExecuteChanged()
        {
            CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        }
    }
}
