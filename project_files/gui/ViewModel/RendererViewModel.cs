using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Controls;
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
            private string m_name = RendererModel.getRendererName(0);
            public int Id { get; set; }
            public Core.RendererType Type
            {
                get => m_type;
                set
                {
                    if (m_type == value) return;
                    m_type = value;
                    m_name = RendererModel.getRendererName(m_type);
                }
            }
            public string Name { get => m_name; }
        }


        private readonly Models m_models;
        private RendererItem m_selectedRenderer;
        private DataGrid m_propertiesGrid;
        

        public RendererItem SelectedRenderer
        {
            get { return m_selectedRenderer; }
            set
            {
                if (m_selectedRenderer == value) return;
                m_selectedRenderer = value;
                Settings.Default.LastSelectedRenderer = m_selectedRenderer.Id;
                m_models.Renderer.Type = m_selectedRenderer.Type;
                OnPropertyChanged(nameof(SelectedRenderer));
            }
        }

        public bool IsRendering
        {
            get => m_models.Renderer.IsRendering;
        }

        public uint Iteration
        {
            get => m_models.Renderer.Iteration;
        }

        public bool AutoStartOnLoad
        {
            get => Settings.Default.AutoStartOnLoad;
            set
            {
                if (Settings.Default.AutoStartOnLoad == value) return;
                Settings.Default.AutoStartOnLoad = value;
                OnPropertyChanged(nameof(AutoStartOnLoad));
            }
        }

        public ObservableCollection<RendererItem> Renderers { get; }
        public ObservableCollection<object> RendererProperties { get; }

        public RendererViewModel(MainWindow window, Models models)
        {
            m_models = models;
            m_propertiesGrid = (DataGrid)((UserControl)window.FindName("RendererPropertiesControl")).FindName("RendererPropertiesGrid");
            RendererProperties = new ObservableCollection<object>();
            AutoStartOnLoad = Settings.Default.AutoStartOnLoad;

            // Enable the renderers (TODO: automatically get them from somewhere?)
            Renderers = new ObservableCollection<RendererItem>()
            {
                new RendererItem{ Id = 0, Type = Core.RendererType.CPU_PT },
            };
            if(Core.mufflon_is_cuda_available())
            {
                Renderers.Add(new RendererItem { Id = 1, Type = Core.RendererType.GPU_PT });
            }

            // Register the handlers
            m_models.Scene.PropertyChanged += sceneChanged;
            m_models.Renderer.PropertyChanged += rendererChanged;

            // Enable the last selected renderer
            int lastSelected = Settings.Default.LastSelectedRenderer;
            if (lastSelected >= Renderers.Count)
            {
                lastSelected = 0;
                Settings.Default.LastSelectedRenderer = 0;
            }
            m_selectedRenderer = Renderers[Settings.Default.LastSelectedRenderer];
            m_models.Renderer.Type = (Core.RendererType)m_selectedRenderer.Type;
        }

        private void sceneChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.Scene.FullPath):
                    if (!Core.render_disable_all_render_targets())
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.render_enable_render_target(Core.RenderTarget.RADIANCE, 0))
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.render_enable_renderer(m_models.Renderer.Type))
                        throw new Exception(Core.core_get_dll_error());
                    if (!Core.render_reset())
                        throw new Exception(Core.core_get_dll_error());
                    if (m_models.Scene.IsLoaded && AutoStartOnLoad)
                        m_models.Renderer.IsRendering = true;
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
                    System.Windows.Application.Current.Dispatcher.Invoke(new Action(() => {
                        OnPropertyChanged(nameof(Iteration));
                    }));
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
