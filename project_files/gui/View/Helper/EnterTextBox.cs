using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;

namespace gui.View.Helper
{
    /// <summary>
    /// extension of the TextBox that loses focus after pressing Enter
    /// </summary>
    public class EnterTextBox : TextBox
    {
        protected override void OnKeyUp(KeyEventArgs e)
        {
            if (e.Key != Key.Enter)
            {
                base.OnKeyUp(e);
                // prevent the key event from bubbling up
                e.Handled = true;
                return;
            }

            // update binding if enter was pressed
            e.Handled = true;
            var binding = BindingOperations.GetBindingExpression(this, TextProperty);
            binding?.UpdateSource();
            Keyboard.ClearFocus();
        }
    }
}
