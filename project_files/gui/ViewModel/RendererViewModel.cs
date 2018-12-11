using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using gui.Annotations;
using gui.Dll;
using gui.Model;
using gui.Properties;

namespace gui.ViewModel
{
    public class RendererViewModel : INotifyPropertyChanged
    {
        public class RendererItem
        {
            public Core.RendererType Type { get; set; }
            public string Name { get; set; }
        }

        private static readonly string LAST_SELECTED_RENDERER_PATH = "LastSelectedRenderer";
        private readonly Models m_models;
        private readonly ObservableCollection<RendererItem> m_renderers = new ObservableCollection<RendererItem>()
        {
            new RendererItem{ Type = Core.RendererType.CPU_PT, Name = "Pathtracer (CPU)" },
            new RendererItem{ Type = Core.RendererType.GPU_PT, Name = "Pathtracer (GPU)" },
        };
        private RendererItem m_selectedRenderer;
        
        public RendererItem SelectedRenderer
        {
            get { return m_selectedRenderer; }
            set
            {
                if (m_selectedRenderer == value) return;
                m_selectedRenderer = value;
                Settings.Default[LAST_SELECTED_RENDERER_PATH] = (int)m_selectedRenderer.Type;
                m_models.Renderer.Type = m_selectedRenderer.Type;
            }
        }
        // TODO: save last selected renderer

        public ObservableCollection<RendererItem> Renderers { get => m_renderers; }

        public RendererViewModel(Models models)
        {
            m_selectedRenderer = m_renderers[(int) Settings.Default[LAST_SELECTED_RENDERER_PATH]];
            m_models = models;
            m_models.Scene.PropertyChanged += sceneChanged;
            m_models.Renderer.PropertyChanged += rendererChanged;
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
                    break;
            }
        }

        private void rendererChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.Renderer.Type):
                    if (!Core.render_disable_all_render_targets())
                        throw new Exception(Core.GetDllError());
                    if (!Core.render_enable_render_target(Core.RenderTarget.RADIANCE, 0))
                        throw new Exception(Core.GetDllError());
                    if (m_models.Scene.IsLoaded)
                    {
                        if (!Core.render_enable_renderer(m_models.Renderer.Type))
                            throw new Exception(Core.GetDllError());
                        if (!Core.render_reset())
                            throw new Exception(Core.GetDllError());
                    }
                    break;
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
}
