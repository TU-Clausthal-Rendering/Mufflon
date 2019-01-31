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
    public class RendererPropertyBool
    {
        public string Name { get; set; }
        public bool Value { get; set; }
    }
    public class RendererPropertyInt
    {
        public string Name { get; set; }
        public int Value { get; set; }
    }
    public class RendererPropertyFloat
    {
        public string Name { get; set; }
        public float Value { get; set; }
    }

    public class RendererViewModel : INotifyPropertyChanged
    {
        // Describes an item in the renderer selection combobox
        public class RendererItem
        {
            private Core.RendererType m_type;
            private string m_name = RendererModel.GetRendererName(0);
            public int Id { get; set; }
            public Core.RendererType Type
            {
                get => m_type;
                set
                {
                    if (m_type == value) return;
                    m_type = value;
                    m_name = RendererModel.GetRendererName(m_type);
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
                m_models.Settings.LastSelectedRenderer = m_selectedRenderer.Id;
                m_models.Renderer.Type = m_selectedRenderer.Type;
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

        public ObservableCollection<RendererItem> Renderers { get; }
        public ObservableCollection<object> RendererProperties { get; }

        public RendererViewModel(Models models, ICommand playPause, ICommand reset)
        {
            m_models = models;
            m_playPause = playPause;
            m_reset = reset;
            m_propertiesGrid = (DataGrid)((UserControl)m_models.App.Window.FindName("RendererPropertiesControl"))?.FindName("RendererPropertiesGrid");
            RendererProperties = new ObservableCollection<object>();

            // Enable the renderers (TODO: automatically get them from somewhere?)
            Array rendererValues = Enum.GetValues(typeof(Core.RendererType));
            Renderers = new ObservableCollection<RendererItem>();
            int typeId = 0;
            foreach (Core.RendererType type in rendererValues)
                if (!Enum.GetName(typeof(Core.RendererType), type).Contains("GPU"))
                    Renderers.Add(new RendererItem { Id = typeId++, Type = type });
            if(Core.mufflon_is_cuda_available())
            {
                foreach (Core.RendererType type in rendererValues)
                    if (Enum.GetName(typeof(Core.RendererType), type).Contains("GPU"))
                        Renderers.Add(new RendererItem { Id = typeId++, Type = type });
            }

            // Register the handlers
            m_models.PropertyChanged += ModelsOnPropertyChanged;
            m_models.Renderer.PropertyChanged += rendererChanged;
            m_models.Settings.PropertyChanged += SettingsOnPropertyChanged;

            // Enable the last selected renderer
            int lastSelected = models.Settings.LastSelectedRenderer;
            if (lastSelected >= Renderers.Count)
            {
                lastSelected = 0;
                models.Settings.LastSelectedRenderer = 0;
            }
            m_selectedRenderer = Renderers[models.Settings.LastSelectedRenderer];
            m_models.Renderer.Type = m_selectedRenderer.Type;
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
                        // TODO other conditions
                        if (!Core.render_disable_all_render_targets())
                            throw new Exception(Core.core_get_dll_error());
                        if (!Core.render_enable_render_target(m_models.RenderTargetSelection.VisibleTarget,
                            m_models.RenderTargetSelection.IsVarianceVisible ? 1u : 0u))
                            throw new Exception(Core.core_get_dll_error());
                        if (!Core.render_enable_renderer(m_models.Renderer.Type))
                            throw new Exception(Core.core_get_dll_error());
                        if (!Core.render_reset())
                            throw new Exception(Core.core_get_dll_error());
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
                case nameof(Models.Renderer.Type):
                    rendererTypeChanged();
                    break;
                case nameof(Models.Renderer.IsRendering):
                    // Update the parameters of the renderer
                    if (m_models.Renderer.IsRendering)
                    {
                        foreach(object prop in RendererProperties) {
                            if(prop is RendererPropertyBool)
                            {
                                if (!Core.renderer_set_parameter_bool((prop as RendererPropertyBool).Name,
                                    Convert.ToUInt32((prop as RendererPropertyBool).Value)))
                                    throw new Exception("Failed to set renderer parameter");
                            } else if(prop is RendererPropertyInt)
                            {
                                if (!Core.renderer_set_parameter_int((prop as RendererPropertyInt).Name,
                                    (prop as RendererPropertyInt).Value))
                                    throw new Exception("Failed to set renderer parameter");
                            } else if (prop is RendererPropertyFloat)
                            {
                                if (!Core.renderer_set_parameter_float((prop as RendererPropertyFloat).Name,
                                    (prop as RendererPropertyFloat).Value))
                                    throw new Exception("Failed to set renderer parameter");
                            }
                        }
                    }
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
            // Perform the actual renderer changes in the DLL
            if (!Core.render_disable_all_render_targets())
                throw new Exception(Core.core_get_dll_error());
            if (!Core.render_enable_render_target(Core.RenderTarget.RADIANCE, 0))
                throw new Exception(Core.core_get_dll_error());
            if (!Core.render_enable_renderer(m_models.Renderer.Type))
                throw new Exception(Core.core_get_dll_error());
            if (!Core.render_reset())
                throw new Exception(Core.core_get_dll_error());

            // Change the properties
            RendererProperties.Clear();
            uint numParams = Core.renderer_get_num_parameters();
            for (uint i = 0; i < numParams; ++i)
            {
                Core.ParameterType type = Core.ParameterType.PARAM_BOOL;
                string name = Core.renderer_get_parameter_desc(i, ref type);
                if (name.Length <= 0)
                    continue;

                switch (type)
                {
                    case Core.ParameterType.PARAM_BOOL:
                        {
                            uint value = 0;
                            if (Core.renderer_get_parameter_bool(name, ref value))
                            {
                                RendererProperties.Add(new RendererPropertyBool()
                                {
                                    Name = name,
                                    Value = value != 0
                                });
                            }
                        }
                        break;
                    case Core.ParameterType.PARAM_INT:
                        {
                            int value = 0;
                            if (Core.renderer_get_parameter_int(name, ref value))
                            {
                                RendererProperties.Add(new RendererPropertyInt()
                                {
                                    Name = name,
                                    Value = value
                                });
                            }
                        }
                        break;
                    case Core.ParameterType.PARAM_FLOAT:
                        {
                            int value = 0;
                            if (Core.renderer_get_parameter_int(name, ref value))
                            {
                                RendererProperties.Add(new RendererPropertyInt()
                                {
                                    Name = name,
                                    Value = value
                                });
                            }
                        }
                        break;
                    default:
                        break;
                }
            }
            OnPropertyChanged(nameof(RendererProperties));
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
