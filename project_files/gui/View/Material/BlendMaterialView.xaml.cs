using System;
using System.Collections.Generic;
using System.ComponentModel;
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
using System.Windows.Navigation;
using System.Windows.Shapes;
using gui.ViewModel.Material;

namespace gui.View.Material
{
    /// <summary>
    /// Interaction logic for BlendMaterialView.xaml
    /// </summary>
    public partial class BlendMaterialView : UserControl
    {
        private readonly BlendMaterialViewModel m_vm;

        public BlendMaterialView(BlendMaterialViewModel vm)
        {
            m_vm = vm;
            InitializeComponent();
            vm.PropertyChanged += VmOnPropertyChanged;
        }

        private void VmOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(BlendMaterialViewModel.LayerA):
                    LayerAHost.Child = m_vm.LayerA as UIElement;
                    break;
                case nameof(BlendMaterialViewModel.LayerB):
                    LayerBHost.Child = m_vm.LayerB as UIElement;
                    break;
            }
        }
    }
}
