using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using gui.Annotations;
using gui.Command;
using gui.Dll;
using gui.Model;

namespace gui.ViewModel
{
    public class TessellationViewModel : INotifyPropertyChanged
    {
        private Models m_models;

        public TessellationViewModel(Models models)
        {
            m_models = models;
            models.PropertyChanged += OnWorldChanged;
            if (m_models.World != null)
                m_models.World.PropertyChanged += OnFrameChanged;

            RequestRetessellationCommand = new ActionCommand(() =>
            {
                if (!Core.scene_request_retessellation())
                    throw new Exception(Core.core_get_dll_error());
            });
        }

        public ICommand RequestRetessellationCommand { get; }

        public uint? MaxTessellationLevel
        {
            get => m_models.World == null ? default(uint?) : m_models.World.MaxTessellationLevel;
            set
            {
                if (m_models.World != null && value.HasValue)
                    m_models.World.MaxTessellationLevel = value.Value;
            }
        }

        private void OnFrameChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.World.MaxTessellationLevel):
                    OnPropertyChanged(nameof(MaxTessellationLevel));
                    break;
            }
        }

        private void OnWorldChanged(object sender, PropertyChangedEventArgs args)
        {
            // Make sure that we notice when a new world has been loaded
            if (args.PropertyName == nameof(Models.World) && m_models.World != null)
            {
                m_models.World.PropertyChanged += OnFrameChanged;
                OnPropertyChanged(nameof(MaxTessellationLevel));
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
