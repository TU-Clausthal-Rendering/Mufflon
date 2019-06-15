using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using gui.Annotations;
using gui.Utility;

namespace gui.Model.Display
{
    public class RenderCameraModel : INotifyPropertyChanged
    {
        public event EventHandler OnMove;

        public float KeyboardSpeed { get; set; } = 0.125f;
        public float MouseSpeed { get; set; } = 0.0025f;

        // Enables realtime-like flight, which hides the cursor and forces uninverted camera control
        private bool m_freeFlightEnabled = false;
        public bool FreeFlightEnabled
        {
            get => m_freeFlightEnabled;
            set
            {
                if (m_freeFlightEnabled == value) return;
                m_freeFlightEnabled = value;
                OnPropertyChanged(nameof(FreeFlightEnabled));
            }
        }

        // Offset incurred by pressed keyboard keys
        public Vec3<float> KeyboardOffset { get; set; } = new Vec3<float>(0f);
        public Vector MouseOffset { get; set; } = new Vector(0, 0);

        // Triggers the OnMove event and resets all offsets
        public void MoveCamera()
        {
            OnMove(this, null);
            KeyboardOffset = new Vec3<float>(0f);
            MouseOffset = new Vector(0, 0);
        }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
