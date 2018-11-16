using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace gui.View.Helper
{
    /// <summary>
    /// helper to make usage of combo boxes easier.
    /// Can hold a any kind of value and a string that will be displayed
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class ComboBoxItem<T>
    {
        private readonly string m_name;

        public T Cargo { get; }

        /// <param name="name">name that will be displayed</param>
        /// <param name="cargo">cargo property</param>
        public ComboBoxItem(string name, T cargo)
        {
            this.m_name = name;
            Cargo = cargo;
        }

        /// <param name="cargo">cargo property and name that will be displayed</param>
        public ComboBoxItem(T cargo)
        {
            m_name = cargo.ToString();
            Cargo = cargo;
        }

        public override string ToString()
        {
            return m_name;
        }
    }
}
