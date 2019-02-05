using System.ComponentModel;
using System.Runtime.CompilerServices;
using gui.Annotations;
using gui.Dll;

namespace gui.Model
{
    public class StatusbarModel : INotifyPropertyChanged
    {
        public ulong CpuTotalMemory { get; private set; }
        public ulong CpuFreeMemory { get; private set; }
        public ulong CpuUsedMemory { get; private set; }
        public ulong CudaTotalMemory { get; private set; }
        public ulong CudaFreeMemory { get; private set; }
        public ulong CudaUsedMemory { get; private set; }

        public void UpdateMemory()
        {
            CpuTotalMemory = Core.profiling_get_total_cpu_memory() / (1024 * 1024);
            CpuFreeMemory = Core.profiling_get_free_cpu_memory() / (1024 * 1024);
            CpuUsedMemory = Core.profiling_get_used_cpu_memory() / (1024 * 1024);
            CudaTotalMemory = Core.profiling_get_total_gpu_memory() / (1024 * 1024);
            CudaFreeMemory = Core.profiling_get_free_gpu_memory() / (1024 * 1024);
            CudaUsedMemory = Core.profiling_get_used_gpu_memory() / (1024 * 1024);
            OnPropertyChanged(nameof(CpuTotalMemory));
            OnPropertyChanged(nameof(CpuFreeMemory));
            OnPropertyChanged(nameof(CpuUsedMemory));
            OnPropertyChanged(nameof(CudaTotalMemory));
            OnPropertyChanged(nameof(CudaFreeMemory));
            OnPropertyChanged(nameof(CudaUsedMemory));
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
