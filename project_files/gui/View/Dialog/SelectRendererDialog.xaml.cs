﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace gui.View.Dialog
{
    /// <summary>
    /// Interaction logic for SelectRendererDialog.xaml
    /// </summary>
    public partial class SelectRendererDialog : Window
    {
        public SelectRendererDialog(object dataContext)
        {
            InitializeComponent();
            DataContext = dataContext;
        }

        private void SelectOnClick(object sender, RoutedEventArgs e)
        {
            DialogResult = true;
            Close();
        }

        private void CancelOnClick(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
