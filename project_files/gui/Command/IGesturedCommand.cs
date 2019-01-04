using System;
using System.Windows;
using System.Windows.Input;
using gui.Properties;

namespace gui.Command
{
    public abstract class IGesturedCommand : ICommand
    {
        public abstract event EventHandler CanExecuteChanged;

        public abstract bool CanExecute(object parameter);
        public abstract void Execute(object parameter);

        private KeyBinding m_keyBind = null;
        private KeyGestureConverter m_converter = new KeyGestureConverter();
        private readonly string PROPERTY_NAME;

        public IGesturedCommand(string propertyName) {
            PROPERTY_NAME = propertyName;
            updateGesture(Settings.Default[PROPERTY_NAME] as string);
        }

        public void updateGesture(string gesture)
        {
            if (m_keyBind != null)
                Application.Current.MainWindow.InputBindings.Remove(m_keyBind);
            Settings.Default[PROPERTY_NAME] = gesture;
            // Set the configured keybinding for the command
            if (gesture != null && gesture.Length != 0)
            {
                m_keyBind = new KeyBinding(this, m_converter.ConvertFromString(gesture as string) as KeyGesture);
                Application.Current.MainWindow.InputBindings.Add(m_keyBind);
            } else
            {
                m_keyBind = null;
            }
        }

        public string getCurrentGesture()
        {
            return Settings.Default[PROPERTY_NAME] as string;
        }

        public string getGestureProperty()
        {
            return PROPERTY_NAME;
        }
    }
}
