using System;
using System.ComponentModel;
using System.Timers;
using System.Runtime.CompilerServices;
using gui.Model;
using gui.Dll;
using gui.Annotations;
using gui.Model.Display;

namespace gui.ViewModel
{
    public class StatusbarViewModel : INotifyPropertyChanged
    {
        private readonly Models m_models;

        public string CpuMemory { get; private set; } = "CPU: NaN/NaN/NaN MB";
        public string CudaMemory { get; private set; } = "CUDA: NaN/NaN/NaN MB";

        public StatusbarViewModel(Models models)
        {
            m_models = models;
            m_models.Statusbar.PropertyChanged += OnMemoryChanged;
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
