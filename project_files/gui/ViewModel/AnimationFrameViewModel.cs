﻿using System.ComponentModel;
using System.Runtime.CompilerServices;
using gui.Annotations;
using gui.Model;

namespace gui.ViewModel
{
    public class AnimationFrameViewModel : INotifyPropertyChanged
    {
        private Models m_models;

        public AnimationFrameViewModel(Models models)
        {
            m_models = models;
            models.PropertyChanged += OnWorldChanged;
            if(m_models.World != null)
                m_models.World.PropertyChanged += OnFrameChanged;
        }

        public uint Current
        {
            get => m_models.World == null ? 0 : m_models.World.AnimationFrameCurrent;
            set
            {
                if(m_models.World != null)
                    m_models.World.AnimationFrameCurrent = value;
            }
        }

        public uint Count { get => m_models.World == null ? 0 : m_models.World.AnimationFrameCount; }
        public uint End { get => Count - 1u; }

        private void OnFrameChanged(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(Models.World.AnimationFrameCount):
                    OnPropertyChanged(nameof(Count));
                    OnPropertyChanged(nameof(End));
                    break;
                case nameof(Models.World.AnimationFrameCurrent):
                    OnPropertyChanged(nameof(Current));
                    break;
            }
        }

        private void OnWorldChanged(object sender, PropertyChangedEventArgs args)
        {
            // Make sure that we notice when a new world has been loaded
            if(args.PropertyName == nameof(Models.World) && m_models.World != null)
            {
                m_models.World.PropertyChanged += OnFrameChanged;
                OnPropertyChanged(nameof(Count));
                OnPropertyChanged(nameof(End));
                OnPropertyChanged(nameof(Current));
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
