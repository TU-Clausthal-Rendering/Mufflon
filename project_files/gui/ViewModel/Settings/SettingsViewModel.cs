using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Annotations;
using gui.Command;
using gui.Model;

namespace gui.ViewModel.Settings
{
    public class SettingsViewModel : INotifyPropertyChanged
    {
        private readonly Models m_models;

        public SettingsViewModel(Models models)
        {
            m_models = models;
            General = new GeneralSettingsViewModel(models);
            Keybindings = new KeybindingsViewModel(models);
            Other = new OtherViewModel(models);

            SaveCommand = new SetDialogResultCommand(m_models, true);
            CancelCommand = new SetDialogResultCommand(m_models, false);
        }

        public void LoadFromSettings()
        {
            General.LoadFromSettings();
            Keybindings.LoadFromSettings();
        }

        public void StoreSettings()
        {
            General.StoreSettings();
            Keybindings.StoreSettings();
        }

        public GeneralSettingsViewModel General { get; }
        public KeybindingsViewModel Keybindings { get; }
        public OtherViewModel Other { get; }

        public ICommand SaveCommand { get; }
        public ICommand CancelCommand { get; }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
