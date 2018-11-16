using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
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
        // subsribe to Models.
        public ObservableCollection<TModel> Models { get; } = new ObservableCollection<TModel>();
    }
}
