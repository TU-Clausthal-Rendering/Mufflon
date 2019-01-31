using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using gui.Dll;
using gui.Model;
using gui.Model.Light;
using gui.View.Dialog;
using gui.ViewModel.Light;

namespace gui.Command
{
    public class AddLightCommand : ICommand
    {
        private readonly Models m_models;

        public AddLightCommand(Models models)
        {
            m_models = models;
            m_models.PropertyChanged += ModelsOnPropertyChanged;
        }

        private void ModelsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.World):
                    OnCanExecuteChanged();
                    break;
            }
        }

        public bool CanExecute(object parameter)
        {
            return m_models.World != null;
        }

        public void Execute(object parameter)
        {
            var dc = new AddLightViewModel();
            var dialog = new AddPropertyDialog(dc);

            if (dialog.ShowDialog() != true) return;

            m_models.World.Lights.AddLight(dc.NameValue, dc.TypeValue);
        }

        public event EventHandler CanExecuteChanged;

        protected virtual void OnCanExecuteChanged()
        {
            CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        }
    }
}
