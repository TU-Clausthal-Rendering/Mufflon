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
    class DecimalTextBox : TextBox
    {
        protected override void OnTextChanged(TextChangedEventArgs e)
        {
            float val;
            if (Text != "")
                if (!float.TryParse(Text, out val))
                    Text = "NaN";
            base.OnTextChanged(e);
        }

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
