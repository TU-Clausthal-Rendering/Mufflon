using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;

namespace gui.View.Helper
{
    /// <summary>
    /// extension of the TextBox that loses focus after pressing Enter
    /// </summary>
    public class RestrictedTextBox : TextBox
    {
        private Regex m_allowed;

        public void changeAllowedRegex(Regex allowed)
        {
            m_allowed = allowed;
        }

        protected override void OnPreviewTextInput(TextCompositionEventArgs e)
        {
            e.Handled = m_allowed.IsMatch(e.Text);
        }

        protected override void OnKeyUp(KeyEventArgs e)
        {
            switch(e.Key)
            {
                case Key.Enter:
                    // update binding if enter was pressed
                    e.Handled = true;
                    var binding = BindingOperations.GetBindingExpression(this, TextProperty);
                    binding?.UpdateSource();
                    Keyboard.ClearFocus();
                    return;
                default:
                    base.OnKeyUp(e);
                    // prevent the key event from bubbling up
                    e.Handled = true;
                    return;
            }
        }

        private void TextBoxPasting(object sender, DataObjectPastingEventArgs e)
        {
            if(e.DataObject.GetDataPresent(typeof(string)))
            {
                string text = (string)e.DataObject.GetData(typeof(string));
                if (!m_allowed.IsMatch(text))
                {
                    e.CancelCommand();
                }
            } else
            {
                e.CancelCommand();
            }
        }
    }
}
