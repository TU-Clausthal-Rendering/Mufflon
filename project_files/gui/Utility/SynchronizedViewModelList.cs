using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace gui.Utility
{
    /// <summary>
    /// helper class to manage a list of models and a list of view models that should be synchronized.
    /// </summary>
    /// <typeparam name="TModel"></typeparam>
    /// <typeparam name="TViewModel"></typeparam>
    /// <typeparam name="TView"></typeparam>
    public abstract class SynchronizedViewModelList<TModel, TViewModel, TView> where TModel : class
    {
        private readonly ObservableCollection<TView> m_views = new ObservableCollection<TView>();
        private readonly List<TViewModel> m_viewModels = new List<TViewModel>();

        public ReadOnlyObservableCollection<TView> Views { get; }

        private SynchronizedModelList<TModel> m_list = null;

        protected SynchronizedViewModelList()
        {
            // wrapper around m_views
            Views = new ReadOnlyObservableCollection<TView>(m_views);
        }

        /// <summary>
        /// Registers (if not null) or unregisters (if null) the new model list that is used for synchronization
        /// </summary>
        /// <param name="modelList"></param>
        protected void RegisterModelList(SynchronizedModelList<TModel> modelList)
        {
            if (m_list != null)
            {
                // unregister old list
                m_list.Models.CollectionChanged -= ModelsOnCollectionChanged;
                m_views.Clear();
            }

            m_list = modelList;
            if(m_list == null) return;

            // register new list
            m_list.Models.CollectionChanged += ModelsOnCollectionChanged;
            // init values
            foreach (var model in m_list.Models)
            {
                var vm = CreateViewModel(model);
                var view = CreateView(vm);
                m_viewModels.Add(vm);
                m_views.Add(view);
            }
        }

        /// <summary>
        /// Creates a new view model from an existing model
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        protected abstract TViewModel CreateViewModel(TModel model);

        /// <summary>
        /// creates a new view from an existing view model
        /// </summary>
        /// <param name="viewModel"></param>
        /// <returns></returns>
        protected abstract TView CreateView(TViewModel viewModel);

        /// <summary>
        /// will be called before the removal of a view and view model from the list
        /// </summary>
        /// <param name="viewModel"></param>
        /// <param name="view"></param>
        protected virtual void OnDeletion(TViewModel viewModel, TView view)
        {

        }

        private void ModelsOnCollectionChanged(object sender, NotifyCollectionChangedEventArgs args)
        {
            switch (args.Action)
            {
                case NotifyCollectionChangedAction.Add:
                    // add new items
                    for (var i = 0; i < args.NewItems.Count; ++i)
                    {
                        var vm = CreateViewModel(args.NewItems[i] as TModel);
                        var view = CreateView(vm);
                        m_viewModels.Insert(args.NewStartingIndex + i, vm);
                        m_views.Insert(args.NewStartingIndex + i, view);
                    }
                    break;
                case NotifyCollectionChangedAction.Remove:
                    // delete old items
                    for (var i = args.OldItems.Count - 1; i >= 0; --i)
                    {
                        OnDeletion(m_viewModels[args.OldStartingIndex + i] ,m_views[args.OldStartingIndex + i]);
                        m_views.RemoveAt(args.OldStartingIndex + i);
                        m_viewModels.RemoveAt(args.OldStartingIndex + i);
                    }
                    break;
                case NotifyCollectionChangedAction.Move:
                case NotifyCollectionChangedAction.Replace:
                    Debug.Assert(false);
                    throw new NotImplementedException("SynchronizedViewModelList Move and Replace action");
                case NotifyCollectionChangedAction.Reset:
                    for (var i = 0; i < m_viewModels.Count; ++i)
                    {
                        OnDeletion(m_viewModels[i], m_views[i]);
                    }
                    m_views.Clear();
                    m_viewModels.Clear();
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }
}
