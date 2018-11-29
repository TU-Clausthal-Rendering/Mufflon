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
        private string m_cpuMemory = "CPU: NaN/NaN/NaN MB";
        private string m_gpuMemory = "GPU: NaN/NaN/NaN MB";
        public string CpuMemory { get => m_cpuMemory; }
        public string GpuMemory { get => m_gpuMemory; }
        private readonly Models m_models;
        private Timer m_timer;

        public StatusbarViewModel(Models models)
        {
            m_models = models;
            updateMemoryDisplay();
            m_timer = new Timer();
            m_timer.Elapsed += new ElapsedEventHandler(updateMemoryTick);
            m_timer.Interval = (2000);
            m_timer.Enabled = true;
            m_timer.Start();
        }

        private void updateMemoryTick(object sender, ElapsedEventArgs e)
        {
            updateMemoryDisplay();
        }

        private void updateMemoryDisplay()
        {
            ulong cpuTotal = Core.profiling_get_total_cpu_memory() / (1024 * 1024);
            ulong cpuFree = Core.profiling_get_free_cpu_memory() / (1024 * 1024);
            ulong cpuUsed = Core.profiling_get_used_cpu_memory() / (1024 * 1024);
            ulong gpuTotal = Core.profiling_get_total_gpu_memory() / (1024 * 1024);
            ulong gpuFree = Core.profiling_get_free_gpu_memory() / (1024 * 1024);
            ulong gpuUsed = Core.profiling_get_used_gpu_memory() / (1024 * 1024);
            m_cpuMemory = "CPU: " + cpuTotal.ToString() + "/" + cpuUsed + "/" + cpuFree + " MB";
            m_gpuMemory = "GPU: " + gpuTotal.ToString() + "/" + gpuUsed + "/" + gpuFree + " MB";
            OnPropertyChanged(nameof(CpuMemory));
            OnPropertyChanged(nameof(GpuMemory));
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
