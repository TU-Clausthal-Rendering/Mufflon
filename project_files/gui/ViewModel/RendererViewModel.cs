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
    // Classes for the different renderer properties
    public class RendererPropertyBool : INotifyPropertyChanged
    {
        private string m_name;
        private bool m_value = false;

        public RendererPropertyBool(string name) { m_name = name; }

        public string Name { get => m_name; }
        public bool Value
        {
            get => m_value;
            set
            {
                if (value == m_value) return;
                m_value = value;
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
        private string m_name;
        private int m_value = 0;

        public RendererPropertyInt(string name) { m_name = name; }

        public string Name { get => m_name; }
        public int Value
        {
            get => m_value;
            set
            {
                if (value == m_value) return;
                m_value = value;
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
        private string m_name;
        private float m_value = 0f;

        public RendererPropertyFloat(string name) { m_name = name; }

        public string Name { get => m_name; }
        public float Value
        {
            get => m_value;
            set
            {
                if (value == m_value) return;
                m_value = value;
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


        private readonly Models m_models;
        private ICommand m_playPause;
        private ICommand m_reset;
        private RendererItem m_selectedRenderer;
        private DataGrid m_propertiesGrid;
        

        public RendererItem SelectedRenderer
        {
            get => m_selectedRenderer;
            set
            {
                if (m_selectedRenderer == value) return;
                m_selectedRenderer = value;
                m_models.Settings.LastSelectedRenderer = m_selectedRenderer.Index;
                m_models.Renderer.RendererIndex = m_selectedRenderer.Index;
                OnPropertyChanged(nameof(SelectedRenderer));
            }
        }

        public bool IsRendering => m_models.Renderer.IsRendering;

        public uint Iteration => m_models.Renderer.Iteration;

        public bool AutoStartOnLoad
        {
            get => m_models.Settings.AutoStartOnLoad;
            set => m_models.Settings.AutoStartOnLoad = value;
        }

        public ObservableCollection<RendererItem> Renderers { get; } = new ObservableCollection<RendererItem>();
        public ObservableCollection<object> RendererProperties { get; } = new ObservableCollection<object>();

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
                    rendererTypeChanged();
                    break;
                case nameof(Models.Renderer.RendererCount):
                {
                    // Enable the renderers
                    Renderers.Clear();
                    for (UInt32 i = 0u; i < m_models.Renderer.RendererCount; ++i)
                    {
                        Renderers.Add(new RendererItem() { Index = i });
                    }
                    // Enable the last selected renderer
                    uint lastSelected = m_models.Settings.LastSelectedRenderer;
                    if (lastSelected >= Renderers.Count)
                    {
                        lastSelected = 0;
                        m_models.Settings.LastSelectedRenderer = 0;
                    }
                        SelectedRenderer = Renderers[(int)m_models.Settings.LastSelectedRenderer];
                    if (m_models.Renderer.RendererIndex == SelectedRenderer.Index)
                        rendererChanged(m_models.Renderer, new PropertyChangedEventArgs(nameof(Models.Renderer.RendererIndex)));
                    else
                        m_models.Renderer.RendererIndex = SelectedRenderer.Index;
                    OnPropertyChanged(nameof(Renderers));
                    }   break;
                case nameof(Models.Renderer.IsRendering):
                    OnPropertyChanged(nameof(IsRendering));
                    break;
                case nameof(Models.Renderer.Iteration):
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() => {
                        OnPropertyChanged(nameof(Iteration));
                    }));
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
            if (m_reset.CanExecute(null))
                m_reset.Execute(null);
            if (!Core.render_enable_renderer(m_models.Renderer.RendererIndex))
                throw new Exception(Core.core_get_dll_error());

            // Change the properties
            RendererProperties.Clear();
            uint numParams = Core.renderer_get_num_parameters();
            for (uint i = 0; i < numParams; ++i)
            {
                Core.ParameterType type;
                string name = Core.renderer_get_parameter_desc(i, out type);
                if (name.Length <= 0)
                    continue;

                switch (type)
                {
                    case Core.ParameterType.Bool:
                        {
                            uint value;
                            if (Core.renderer_get_parameter_bool(name, out value))
                            {
                                var prop = new RendererPropertyBool(name) { Value = value != 0u };
                                prop.PropertyChanged += OnRenderPropertyChanged;
                                RendererProperties.Add(prop);
                            }
                        }
                        break;
                    case Core.ParameterType.Int:
                        {
                            int value;
                            if (Core.renderer_get_parameter_int(name, out value))
                            {
                                var prop = new RendererPropertyInt(name) { Value = value };
                                prop.PropertyChanged += OnRenderPropertyChanged;
                                RendererProperties.Add(prop);
                            }
                        }
                        break;
                    case Core.ParameterType.Float:
                        {
                            float value;
                            if (Core.renderer_get_parameter_float(name, out value))
                            {
                                var prop = new RendererPropertyFloat(name) { Value = value };
                                prop.PropertyChanged += OnRenderPropertyChanged;
                                RendererProperties.Add(prop);
                            }
                        }
                        break;
                    default:
                        break;
                }
            }
            OnPropertyChanged(nameof(RendererProperties));
        }

        private void OnRenderPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            if (sender is RendererPropertyBool)
            {
                if (!Core.renderer_set_parameter_bool((sender as RendererPropertyBool).Name,
                    Convert.ToUInt32((sender as RendererPropertyBool).Value)))
                    throw new Exception("Failed to set renderer parameter");
            }
            else if (sender is RendererPropertyInt)
            {
                if (!Core.renderer_set_parameter_int((sender as RendererPropertyInt).Name,
                    (sender as RendererPropertyInt).Value))
                    throw new Exception("Failed to set renderer parameter");
            }
            else if (sender is RendererPropertyFloat)
            {
                if (!Core.renderer_set_parameter_float((sender as RendererPropertyFloat).Name,
                    (sender as RendererPropertyFloat).Value))
                    throw new Exception("Failed to set renderer parameter");
            }

            if (m_reset.CanExecute(null))
                m_reset.Execute(null);
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
