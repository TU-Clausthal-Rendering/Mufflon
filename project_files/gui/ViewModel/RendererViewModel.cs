using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Controls;
using System.Windows.Input;
using gui.Annotations;
using gui.Dll;
using gui.Model;
using gui.Properties;

namespace gui.ViewModel
{
    // https://stackoverflow.com/a/17142844/1913512
    /*public class FloatConverter : System.Windows.Data.IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            // return an invalid value in case of the value ends with a point
            //return value.ToString().EndsWith(".") ? "." : value;
            value = System.Text.RegularExpressions.Regex.Replace(value.ToString(), "[^.0-9]", "");
            return value;
        }
    }*/

    // Classes for the different renderer properties
    public class RendererPropertyBool : INotifyPropertyChanged
    {
        private RendererParameter m_param;

        public RendererPropertyBool(RendererParameter param) { m_param = param; }

        public string Name { get => m_param.Name; }
        public bool Value
        {
            get => (bool) m_param.Value;
            set
            {
                if (value == (bool)m_param.Value) return;
                m_param.Value = value;
                OnPropertyChanged(nameof(Value));
            }
        }

        #region PropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
    public class RendererPropertyInt : INotifyPropertyChanged
    {
        private RendererParameter m_param;

        public RendererPropertyInt(RendererParameter param) { m_param = param; }

        public string Name { get => m_param.Name; }
        public int Value
        {
            get => (int)m_param.Value;
            set
            {
                if (value == (int)m_param.Value) return;
                m_param.Value = value;
                OnPropertyChanged(nameof(Value));
            }
        }

        #region PropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
    public class RendererPropertyFloat : INotifyPropertyChanged
    {
        private RendererParameter m_param;

        public RendererPropertyFloat(RendererParameter param) { m_param = param; }

        public string Name { get => m_param.Name; }
        public float Value
        {
            get => (float)m_param.Value;
            set
            {
                if (value == (float)m_param.Value) return;
                m_param.Value = value;
                OnPropertyChanged(nameof(Value));
            }
        }

        #region PropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
    public class RendererPropertyEnum : INotifyPropertyChanged
    {
        private RendererParameter m_param;

        public class Item
        {
            private uint m_index = UInt32.MaxValue;
            private string m_propName;
            private string m_name;
            private int m_value;

            public Item(uint index, string propName)
            {
                m_propName = propName;
                Index = index;
            }

            public uint Index
            {
                get => m_index;
                set
                {
                    if (m_index == value) return;
                    m_index = value;
                    if(!Core.renderer_get_parameter_enum_value_from_index(m_propName, m_index, out m_value))
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.renderer_get_parameter_enum_name(m_propName, m_value, out m_name))
                        throw new Exception(Core.core_get_dll_error());
                }
            }
            public string Name { get => m_name; }
            public int Value { get => m_value; }
        }

        private Item m_selectedValue;

        public RendererPropertyEnum(RendererParameter param) {
            m_param = param;
            uint num;
            if (!Core.renderer_get_parameter_enum_count(m_param.Name, out num))
                throw new Exception(Core.core_get_dll_error());
            for (uint i = 0u; i < num; ++i)
                Values.Add(new Item(i, m_param.Name));
            SelectedValue = get_index(param.Value as string);
        }

        public Item get_index(string name)
        {
            // TODO: this is kind of inefficient, but I didn't want to introduce another map to the enum parameter...
            int value;
            uint index;
            if(!Core.renderer_get_parameter_enum_value_from_name(m_param.Name, name, out value))
                throw new Exception(Core.core_get_dll_error());
            if (!Core.renderer_get_parameter_enum_index_from_value(m_param.Name, value, out index))
                throw new Exception(Core.core_get_dll_error());
            return Values[(int)index];
        }

        public string Name { get => m_param.Name; }

        public ObservableCollection<Item> Values { get; } = new ObservableCollection<Item>();

        public Item SelectedValue
        {
            get => m_selectedValue;
            set
            {
                if (value == m_selectedValue) return;
                m_selectedValue = value;
                if(m_selectedValue != null)
                    m_param.Value = m_selectedValue.Name;
                OnPropertyChanged(nameof(SelectedValue));
            }
        }

        #region PropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }

    public class RendererViewModel : INotifyPropertyChanged
    {
        // Describes an item in the renderer selection combobox
        public class RendererItem
        {
            private UInt32 m_index = UInt32.MaxValue;
            private string m_name;
            public UInt32 Index
            {
                get => m_index;
                set
                {
                    if (m_index == value) return;
                    m_index = value;
                    m_name = Core.render_get_renderer_name(m_index);
                }
            }
            public string Name { get => m_name; }
        }

        public class RendererVariation
        {
            private uint m_index = 0;
            private uint m_variation = uint.MaxValue;
            private string m_name;

            public RendererVariation(uint index)
            {
                m_index = index;
            }

            public UInt32 Variation
            {
                get => m_variation;
                set
                {
                    if (m_variation == value) return;
                    m_variation = value;
                    m_name = "";
                    Core.RenderDevice devices = Core.render_get_renderer_devices(m_index, m_variation);
                    if (devices == Core.RenderDevice.None)
                    {
                        m_name = "None";
                    } else
                    {
                        if ((devices & Core.RenderDevice.Cpu) != 0)
                            m_name += "CPU, ";
                        if ((devices & Core.RenderDevice.Cuda) != 0)
                            m_name += "CUDA, ";
                        if ((devices & Core.RenderDevice.OpenGL) != 0)
                            m_name += "OPENGL, ";
                        m_name = m_name.Substring(0, m_name.Length - 2);
                    }
                    RenderDevices = devices;
                }
            }

            public Core.RenderDevice RenderDevices { get; private set; }

            public string Name { get => m_name; }
        }


        private readonly Models m_models;
        private ICommand m_playPause;
        private ICommand m_reset;
        private RendererItem m_selectedRenderer;
        private RendererVariation m_selectedRendererVariation;
        private DataGrid m_propertiesGrid;
        

        public RendererItem SelectedRenderer
        {
            get => m_selectedRenderer;
            set
            {
                if (m_selectedRenderer == value) return;
                m_selectedRenderer = value;
                // Save the used devices
                var usedDevices = Core.RenderDevice.None;
                if(m_models.Renderer.RendererIndex < uint.MaxValue)
                    usedDevices = m_models.Renderer.RenderDevices;
                m_models.Settings.LastSelectedRenderer = m_selectedRenderer.Index;
                // Reset the variation to first
                m_models.Renderer.SetRenderer(m_selectedRenderer.Index, 0);
                // The variants get updated in rendererChanged
                // Restore the variant with same devices if applicable
                foreach(var variant in SupportedRenderVariations)
                {
                    if(variant.RenderDevices == usedDevices)
                    {
                        SelectedRendererVariation = variant;
                        break;
                    }
                }

                OnPropertyChanged(nameof(SelectedRenderer));
            }
        }
        public RendererVariation SelectedRendererVariation
        {
            get => m_selectedRendererVariation;
            set
            {
                if (m_selectedRendererVariation == value || value == null) return;
                m_selectedRendererVariation = value;
                m_models.Settings.LastSelectedRendererVariation = m_selectedRendererVariation.Variation;
                m_models.Renderer.SetRenderer(m_models.Renderer.RendererIndex, m_selectedRendererVariation.Variation);
                OnPropertyChanged(nameof(SelectedRendererVariation));
            }
        }

        public bool IsRendering => m_models.Renderer.IsRendering;

        public uint Iteration => m_models.Renderer.Iteration;

        public string CurrentIterationTime { get; private set; } = "0s / 0Gcyc";
        public string AverageIterationTime { get; private set; } = "0s / 0Gcyc";
        public string TotalIterationTime { get; private set; } = "0s / 0Gcyc";

        public bool AutoStartOnLoad
        {
            get => m_models.Settings.AutoStartOnLoad;
            set => m_models.Settings.AutoStartOnLoad = value;
        }

        public ObservableCollection<RendererItem> Renderers { get; } = new ObservableCollection<RendererItem>();
        public ObservableCollection<RendererVariation> SupportedRenderVariations{ get; } = new ObservableCollection<RendererVariation>();
        public ObservableCollection<INotifyPropertyChanged> RendererProperties { get; } = new ObservableCollection<INotifyPropertyChanged>();

        public RendererViewModel(Models models, ICommand playPause, ICommand reset)
        {
            m_models = models;
            m_playPause = playPause;
            m_reset = reset;
            m_propertiesGrid = (DataGrid)((UserControl)m_models.App.Window.FindName("RendererPropertiesControl"))?.FindName("RendererPropertiesGrid");

            // Register the handlers
            m_models.PropertyChanged += ModelsOnPropertyChanged;
            m_models.Renderer.PropertyChanged += rendererChanged;
            m_models.Settings.PropertyChanged += SettingsOnPropertyChanged;
            m_models.App.Loaded += OnFinishedLoading;

            // Renderer initialization is deferred until the render DLL is initialized
        }

        private void SettingsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(SettingsModel.AutoStartOnLoad):
                    OnPropertyChanged(nameof(AutoStartOnLoad));
                    break;
            }
        }

        private void ModelsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.World):
                    if (m_models.World != null)
                    {
                        if (m_reset.CanExecute(null))
                            m_reset.Execute(null);
                        if (AutoStartOnLoad && m_playPause.CanExecute(null))
                           m_playPause.Execute(null);
                    }
                    break;
            }
        }

        private void rendererChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.Renderer.RendererIndex):
                    SupportedRenderVariations.Clear();
                    var variationCount = m_models.Renderer.RendererVariationsCount;
                    for (uint v = 0u; v < variationCount; ++v)
                        SupportedRenderVariations.Add(new RendererVariation(m_models.Renderer.RendererIndex) { Variation = v });
                    SelectedRendererVariation = SupportedRenderVariations[0];
                    OnPropertyChanged(nameof(SupportedRenderVariations));
                    rendererTypeChanged();
                    break;
                case nameof(Models.Renderer.RendererVariation):
                    rendererTypeChanged();
                    break;
                case nameof(Models.Renderer.IsRendering):
                    OnPropertyChanged(nameof(IsRendering));
                    break;
                case nameof(Models.Renderer.Iteration):
                    System.Windows.Application.Current.Dispatcher.Invoke(new Action(() => {
                        OnPropertyChanged(nameof(Iteration));
                    }));
                    break;
                case nameof(Models.Renderer.CurrentIterationTime):
                    CurrentIterationTime = (m_models.Renderer.CurrentIterationTime.microseconds / 1_000_000f).ToString("0.###")
                        + "s / " + (m_models.Renderer.CurrentIterationTime.cycles / 1_000_000_000f).ToString("0.###") + "Gcyc";
                    OnPropertyChanged(nameof(CurrentIterationTime));
                    break;
                case nameof(Models.Renderer.AverageIterationTime):
                    AverageIterationTime = (m_models.Renderer.AverageIterationTime.microseconds / 1_000_000f).ToString("0.###")
                        + "s / " + (m_models.Renderer.AverageIterationTime.cycles / 1_000_000_000f).ToString("0.###") + "Gcyc";
                    OnPropertyChanged(nameof(AverageIterationTime));
                    break;
                case nameof(Models.Renderer.TotalIterationTime):
                    TotalIterationTime = (m_models.Renderer.TotalIterationTime.microseconds / 1_000_000f).ToString("0.###")
                        + "s / " + (m_models.Renderer.TotalIterationTime.cycles / 1_000_000_000f).ToString("0.###") + "Gcyc";
                    OnPropertyChanged(nameof(TotalIterationTime));
                    break;
                case nameof(Models.RenderTargetSelection.VisibleTarget):
                case nameof(Models.RenderTargetSelection.IsVarianceVisible):
                    if (m_reset.CanExecute(null))
                        m_reset.Execute(null);
                    break;
            }
        }

        private void rendererTypeChanged()
        {
            // Change the properties
            RendererProperties.Clear();

            foreach(RendererParameter param in m_models.Renderer.Parameters)
            {
                param.PropertyChanged -= OnRenderParameterChanged;
                switch (param.Type)
                {
                    case Core.ParameterType.Bool:
                        RendererProperties.Add(new RendererPropertyBool(param));
                        break;
                    case Core.ParameterType.Int:
                        RendererProperties.Add(new RendererPropertyInt(param));
                        break;
                    case Core.ParameterType.Float:
                        RendererProperties.Add(new RendererPropertyFloat(param));
                        break;
                    case Core.ParameterType.Enum:
                        RendererProperties.Add(new RendererPropertyEnum(param));
                        break;
                    default:
                        throw new Exception("Invalid renderer parameter type!");
                }
                param.PropertyChanged += OnRenderParameterChanged;
            }

            OnPropertyChanged(nameof(RendererProperties));
        }

        private void OnRenderParameterChanged(object sender, PropertyChangedEventArgs args)
        {
            var param = sender as RendererParameter;
            var prop = RendererProperties[(int)param.Index];

            if (prop is RendererPropertyBool)
                (prop as RendererPropertyBool).Value = (bool)param.Value;
            else if (prop is RendererPropertyInt)
                (prop as RendererPropertyInt).Value = (int)param.Value;
            else if (prop is RendererPropertyEnum)
                (prop as RendererPropertyEnum).SelectedValue = (prop as RendererPropertyEnum).get_index(param.Value as string);
            else if (prop is RendererPropertyFloat)
                (prop as RendererPropertyFloat).Value = (float)param.Value;
        }

        // We may now obtain the proper renderers
        private void OnFinishedLoading(object sender, EventArgs args)
        {
            // Enable the renderers
            Renderers.Clear();
            SupportedRenderVariations.Clear();
            var rendererCount = m_models.Renderer.RendererCount;
            for (UInt32 i = 0u; i < rendererCount; ++i)
                Renderers.Add(new RendererItem() { Index = i });
            // Enable the last selected renderer
            uint lastSelected = m_models.Settings.LastSelectedRenderer;
            uint lastSelectedVariation = m_models.Settings.LastSelectedRendererVariation;
            if (lastSelected >= Renderers.Count)
            {
                lastSelected = 0;
                m_models.Settings.LastSelectedRenderer = 0;
            }
            SelectedRenderer = Renderers[(int)lastSelected];

            if (m_models.Renderer.RendererIndex == SelectedRenderer.Index)
                rendererChanged(m_models.Renderer, new PropertyChangedEventArgs(nameof(Models.Renderer.RendererIndex)));
            else
                m_models.Renderer.SetRenderer(m_selectedRenderer.Index, 0);

            // Set the active variant (the list has been build already)
            if (lastSelectedVariation >= SupportedRenderVariations.Count)
            {
                lastSelectedVariation = 0;
                m_models.Settings.LastSelectedRendererVariation = 0;
            }
            SelectedRendererVariation = SupportedRenderVariations[(int)lastSelectedVariation];


            // Load the parameter values from settings
            OnPropertyChanged(nameof(Renderers));
        }

        #region PropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
}
