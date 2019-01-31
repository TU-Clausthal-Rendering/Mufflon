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
    public class SynchronizedModelList<TModel>
    {
        protected readonly ObservableCollection<TModel> m_list;

        // expose readonly list
        public IReadOnlyCollection<TModel> Models { get; }

        // forward event
        public event NotifyCollectionChangedEventHandler CollectionChanged
        {
            add => m_list.CollectionChanged += value;
            remove => m_list.CollectionChanged -= value;
        }

        public SynchronizedModelList()
        {
            m_list = new ObservableCollection<TModel>();
            Models = new ReadOnlyObservableCollection<TModel>(m_list);
        }
    }
}
