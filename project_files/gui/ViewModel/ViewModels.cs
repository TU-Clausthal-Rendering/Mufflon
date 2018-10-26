using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using gui.Annotations;
using gui.Model;

namespace gui.ViewModel
{
    /// <summary>
    /// class containing all static view models
    /// </summary>
    public class ViewModels
    {
        public ConsoleViewModel Console { get; }

        private readonly Models m_models;

        public ViewModels(MainWindow window)
        {
            // model initialization
            m_models = new Models(window);

            // view model initialization
            Console = new ConsoleViewModel(m_models);

            // command initialization
        }
    }
}
