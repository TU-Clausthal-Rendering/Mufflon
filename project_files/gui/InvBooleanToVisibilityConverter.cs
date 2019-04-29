using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace gui
{
    [ValueConversion(typeof(Visibility), typeof(Visibility))]
    public class InvBooleanToVisibilityConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            bool val = (bool)value;
            if (val)
                return Visibility.Collapsed;
            else
                return Visibility.Visible;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            Visibility vis = (Visibility)value;
            if (vis == Visibility.Collapsed)
                return true;
            else
                return false;
        }
    }
}
