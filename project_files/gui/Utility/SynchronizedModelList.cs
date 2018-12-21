using gui.Annotations;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace gui.Utility
{
    /// <summary>
    /// helper class to manage a list of models and a list of view models that should be synchronized.
    /// </summary>
    /// <typeparam name="TModel"></typeparam>
    public class SynchronizedModelList<TModel> : INotifyPropertyChanged where TModel : class
    {
        // subsribe to Models.
        public ObservableCollection<TModel> Models { get; } = new ObservableCollection<TModel>();
        public ObservableCollection<TModel> RemovedModels { get; } = new ObservableCollection<TModel>();

        public SynchronizedModelList() {
            Models.CollectionChanged += OnModelChanged;
        }

        private void OnModelChanged(object sender, NotifyCollectionChangedEventArgs args)
        {
            if(args.Action == NotifyCollectionChangedAction.Remove)
            {
                RemovedModels.Add(args.OldItems[0] as TModel);
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
