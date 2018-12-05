using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using gui.Annotations;
using gui.Dll;
using gui.Model;

namespace gui.ViewModel
{
    public class RendererViewModel : INotifyPropertyChanged
    {
        private readonly Models m_models;

        public RendererViewModel(Models models)
        {
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
                    if (!Core.render_enable_renderer(m_models.Renderer.Type))
                        throw new Exception(Core.GetDllError());
                    if(!Core.render_reset())
                        throw new Exception(Core.GetDllError());
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
