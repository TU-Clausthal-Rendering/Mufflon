using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace gui.View.Helper
{
    /// <summary>
    /// specialozation of ComboBoxItem that converts a "CamelCase" enum to a "Camel Case" name
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class EnumBoxItem<T> : ComboBoxItem<T> where T : Enum
    {
        public EnumBoxItem(T cargo) : base(EnumToString(cargo), cargo)
        {

        }

        /// <summary>
        /// creates an observable collection with all enums
        /// </summary>
        /// <returns></returns>
        public static ObservableCollection<EnumBoxItem<T>> MakeCollection()
        {
            var res = new ObservableCollection<EnumBoxItem<T>>();
            foreach (var e in Enum.GetValues(typeof(T)))
            {
                res.Add(new EnumBoxItem<T>((T) e));
            }

            return res;
        }

        /// <summary>
        /// converts CamelCase enum to Camel Case
        /// </summary>
        /// <param name="e"></param>
        /// <returns></returns>
        private static string EnumToString(T e)
        {
            var res = "";
            foreach (var letter in e.ToString())
            {
                res += letter;
                // add space if uppercase
                if (char.IsUpper(letter) && res.Length != 1)
                    res += " ";
            }

            return res;
        }
    }
}
