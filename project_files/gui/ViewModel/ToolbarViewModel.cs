using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Input;
using gui.Annotations;
using gui.Command;
using gui.Model;

namespace gui.ViewModel
{
    public class ToolbarViewModel : INotifyPropertyChanged
    {
        private readonly Models m_models;

        public ToolbarViewModel(Models models)
        {
            m_models = models;

            PlayPauseCommand = new PlayPauseCommand(models);
            ResetCommand = new ResetCommand(models, PlayPauseCommand);
            SaveScreenShotCommand = new ScreenShotCommand(m_models);

            m_models.Renderer.PropertyChanged += RendererOnPropertyChanged;
        }

        private void RendererOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(RendererModel.IsRendering):
                    OnPropertyChanged(nameof(PauseIconVisibility));
                    OnPropertyChanged(nameof(PlayIconVisibility));
                    break;
            }
        }

        public Visibility PlayIconVisibility =>
            m_models.Renderer.IsRendering ? Visibility.Collapsed : Visibility.Visible;

        public Visibility PauseIconVisibility =>
            m_models.Renderer.IsRendering ? Visibility.Visible : Visibility.Collapsed;

        public IGesturedCommand PlayPauseCommand { get; }
        public IGesturedCommand ResetCommand { get; }
        public IGesturedCommand SaveScreenShotCommand { get; }
        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
