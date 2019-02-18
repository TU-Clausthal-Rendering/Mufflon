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
                    var devices = Enum.GetValues(typeof(Core.RenderDevice));
                    int count = 0;
                    foreach(Core.RenderDevice device in devices)
                    {
                        if(Core.render_renderer_uses_device(m_index, device))
                        {
                            if (count++ == 0)
                                m_name += " (";
                            else
                                m_name += ", ";
                            m_name += Enum.GetName(typeof(Core.RenderDevice), device);
                        }
                    }
                    if (count > 0)
                        m_name += ")";
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

                    // Load the parameter values from settings
                    

                    OnPropertyChanged(nameof(Renderers));
                    }   break;
                case nameof(Models.Renderer.IsRendering):
                    OnPropertyChanged(nameof(IsRendering));
                    break;
                case nameof(Models.Renderer.Iteration):
                    System.Windows.Application.Current.Dispatcher.Invoke(new Action(() => {
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
            // Change the properties
            RendererProperties.Clear();

            foreach(RendererParameter param in m_models.Renderer.Parameters)
            {
                switch(param.Type)
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
            else if (prop is RendererPropertyFloat)
                (prop as RendererPropertyFloat).Value = (float)param.Value;
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
