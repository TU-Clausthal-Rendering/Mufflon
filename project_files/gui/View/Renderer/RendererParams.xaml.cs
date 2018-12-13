using gui.ViewModel;
using System;
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
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace gui.View.Renderer
{
    public class ColumnTemplateSelector : DataTemplateSelector
    {
        public DataTemplate BoolTemplate { get; set; }
        public DataTemplate IntTemplate { get; set; }
        public DataTemplate FloatTemplate { get; set; }

        public override DataTemplate SelectTemplate(object item, DependencyObject container)
        {
            if (item is RendererPropertyBool)
                return BoolTemplate;
            else if (item is RendererPropertyInt)
                return IntTemplate;
            else if (item is RendererPropertyFloat)
                return FloatTemplate;
            else
                return base.SelectTemplate(item, container);
        }
    }

    /// <summary>
    /// Interaction logic for RendererParams.xaml
    /// </summary>
    public partial class RendererParams : UserControl
    {
        public RendererParams()
        {
            InitializeComponent();
        }
    }
}
