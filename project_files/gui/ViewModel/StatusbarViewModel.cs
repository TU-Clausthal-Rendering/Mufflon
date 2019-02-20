using System;
using System.ComponentModel;
using System.Timers;
using System.Runtime.CompilerServices;
using gui.Model;
using gui.Dll;
using gui.Annotations;

namespace gui.ViewModel
{
    public class StatusbarViewModel : INotifyPropertyChanged
    {
        private readonly Models m_models;

        public string CpuMemory { get; private set; } = "CPU: NaN/NaN/NaN MB";
        public string CudaMemory { get; private set; } = "CUDA: NaN/NaN/NaN MB";

        public string CursorPos { get; private set; } = "0, 0";

        public StatusbarViewModel(Models models)
        {
            m_models = models;
            m_models.Statusbar.PropertyChanged += OnMemoryChanged;
            m_models.Viewport.PropertyChanged += OnCursorChanged;
        }

        private void OnMemoryChanged(object sender, PropertyChangedEventArgs args)
        {
            CpuMemory = "CPU: " + m_models.Statusbar.CpuTotalMemory.ToString() + "/"
                + m_models.Statusbar.CpuUsedMemory + "/"
                + m_models.Statusbar.CpuFreeMemory + " MB";
            CudaMemory = "CUDA: " + m_models.Statusbar.CudaTotalMemory.ToString() + "/"
                + m_models.Statusbar.CudaUsedMemory + "/"
                + m_models.Statusbar.CudaFreeMemory + " MB";
            OnPropertyChanged(nameof(CpuMemory));
            OnPropertyChanged(nameof(CudaMemory));
        }

        private void OnCursorChanged(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(ViewportModel.CursorPosX):
                case nameof(ViewportModel.CursorPosY):
                    CursorPos = m_models.Viewport.CursorPosX.ToString() + ", " + m_models.Viewport.CursorPosY.ToString();
                    OnPropertyChanged(nameof(CursorPos));
                    break;
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
