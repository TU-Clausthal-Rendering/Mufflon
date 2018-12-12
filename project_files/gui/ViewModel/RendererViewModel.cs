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
    public class RendererPropertyString
    {
        public string Name { get; set; }
        public string Value { get; set; }
    }

    public class RendererViewModel : INotifyPropertyChanged
    {
        // Describes an item in the renderer selection combobox
        public class RendererItem
        {
            public int Id { get; set; }
            public Core.RendererType Type { get; set; }
            public string Name { get; set; }
        }


        private readonly Models m_models;
        private readonly ObservableCollection<RendererItem> m_renderers = new ObservableCollection<RendererItem>()
        {
            new RendererItem{ Id = 0, Type = Core.RendererType.CPU_PT, Name = "Pathtracer (CPU)" },
            new RendererItem{ Id = 1, Type = Core.RendererType.GPU_PT, Name = "Pathtracer (GPU)" },
        };
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

        public bool IsRendering {
            get => m_models.Renderer.IsRendering;
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

        public ObservableCollection<RendererItem> Renderers { get => m_renderers; }
        public ObservableCollection<object> RendererProperties { get; }

        public RendererViewModel(MainWindow window, Models models)
        {
            m_models = models;
            m_propertiesGrid = (DataGrid)((UserControl)window.FindName("RendererPropertiesControl")).FindName("RendererPropertiesGrid");
            RendererProperties = new ObservableCollection<object>();
            RendererProperties.Add(new RendererPropertyBool() { Name = "TestBool", Value = true });
            RendererProperties.Add(new RendererPropertyInt() { Name = "TestInt", Value = 6 });
            RendererProperties.Add(new RendererPropertyString() { Name = "TestString", Value = "Ha!" });
            AutoStartOnLoad = Settings.Default.AutoStartOnLoad;

            // Register the handlers
            m_models.Scene.PropertyChanged += sceneChanged;
            m_models.Renderer.PropertyChanged += rendererChanged;

            // Enable the last selected renderer
            int lastSelected = Settings.Default.LastSelectedRenderer;
            if (lastSelected >= m_renderers.Count)
            {
                lastSelected = 0;
                Settings.Default.LastSelectedRenderer = 0;
            }
            m_selectedRenderer = m_renderers[Settings.Default.LastSelectedRenderer];
            m_models.Renderer.Type = (Core.RendererType)m_selectedRenderer.Type;
        }

        private void sceneChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.Scene.FullPath):
                    if (!Core.render_disable_all_render_targets())
                        throw new Exception(Core.GetDllError());
                    if (!Core.render_enable_render_target(Core.RenderTarget.RADIANCE, 0))
                        throw new Exception(Core.GetDllError());
                    if (!Core.render_enable_renderer(m_models.Renderer.Type))
                        throw new Exception(Core.GetDllError());
                    if (!Core.render_reset())
                        throw new Exception(Core.GetDllError());
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
                    OnPropertyChanged(nameof(IsRendering));
                    break;
            }
        }

        private void rendererTypeChanged()
        {
            // Perform the actual renderer changes in the DLL
            if (!Core.render_disable_all_render_targets())
                throw new Exception(Core.GetDllError());
            if (!Core.render_enable_render_target(Core.RenderTarget.RADIANCE, 0))
                throw new Exception(Core.GetDllError());
            if (!Core.render_enable_renderer(m_models.Renderer.Type))
                throw new Exception(Core.GetDllError());
            if (!Core.render_reset())
                throw new Exception(Core.GetDllError());

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
                        {
                            RendererProperties.Add(new RendererPropertyString()
                            {
                                Name = name,
                                Value = "## UNKNOWN TYPE ##"
                            });
                        }
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
