using gui.Dll;
using gui.Model;
using System;
using System.ComponentModel;
using System.Windows.Input;

namespace gui.Command
{
    public class RenderSequenceCommand : ICommand
    {
        private readonly Models m_models;

        int m_start, m_end, m_initial;
        bool m_takeScreenshots;

        public RenderSequenceCommand(Models models, int start, int end, int initial, bool takeScreenshots)
        {
            m_models = models;
            m_models.PropertyChanged += ModelsOnPropertyChanged;
            m_models.Renderer.PropertyChanged += RendererOnPropertyChanged;
            m_models.Toolbar.PropertyChanged += IterationsOnPropertyChanged;
            m_start = start;
            m_end = end;
            m_initial = initial;
            m_takeScreenshots = takeScreenshots;
        }

        private void ModelsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            if (args.PropertyName == nameof(Models.World))
                OnCanExecuteChanged();
        }

        private void RendererOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            if (sender == m_models.Renderer && args.PropertyName == nameof(Models.Renderer.IsRendering))
                OnCanExecuteChanged();
        }

        private void IterationsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            if (sender == m_models.Toolbar && args.PropertyName == nameof(Models.Toolbar.Iterations))
                OnCanExecuteChanged();
        }

        public bool CanExecute(object parameter)
        {
            return m_models.World != null && !m_models.Renderer.IsRendering
                && m_models.Toolbar.Iterations.HasValue;
        }

        public void Execute(object parameter)
        {
            // Initiate a sequence rendering
            if (m_initial >= 0)
                m_models.World.AnimationFrameCurrent = (uint)Math.Max(m_models.World.AnimationFrameStart,
                                                                      Math.Min(m_models.World.AnimationFrameEnd, m_initial));
            if (m_start >= 0)
                m_models.Renderer.AnimationStart = (int)Math.Max(m_models.World.AnimationFrameStart,
                                                                  Math.Min(m_models.World.AnimationFrameEnd, m_start));
            else
                m_models.Renderer.AnimationStart = -1;
            if (m_end >= 0)
                m_models.Renderer.AnimationEnd = (int)Math.Max(m_models.World.AnimationFrameStart,
                                                                  Math.Min(m_models.World.AnimationFrameEnd, m_end));
            else
                m_models.Renderer.AnimationEnd = -1;

            if(m_takeScreenshots)
            {
                if(!m_models.Settings.ScreenshotNamePattern.Contains("#frame"))
                {
                    Logger.log("Screenshot name does not contain '#frame'; all rendered animated screenshots would be overwritten!",
                        Core.Severity.Error);
                    return;
                }
                m_models.Renderer.AnimationFrameComplete = () =>
                {
                    m_models.Renderer.TakeScreenshot(false);
                };
            }

            m_models.Renderer.RenderAnimation = true;
            m_models.Renderer.Iterate(m_models.Toolbar.Iterations.Value);
        }

        public event EventHandler CanExecuteChanged;

        protected virtual void OnCanExecuteChanged()
        {
            CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        }
    }
}
