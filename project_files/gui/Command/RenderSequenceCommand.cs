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

        int m_initialFrame;
        bool m_loopAnimation;
        bool m_takeScreenshots;

        public RenderSequenceCommand(Models models, int initialFrame, bool loopAnimation, bool takeScreenshots)
        {
            m_models = models;
            m_models.PropertyChanged += ModelsOnPropertyChanged;
            m_models.Renderer.PropertyChanged += RendererOnPropertyChanged;
            m_models.Toolbar.PropertyChanged += IterationsOnPropertyChanged;
            m_initialFrame = initialFrame;
            m_loopAnimation = loopAnimation;
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
            if (m_initialFrame >= 0 && m_models.World.AnimationFrameCount != 0u)
                m_models.World.AnimationFrameCurrent = (uint)Math.Min(m_initialFrame, m_models.World.AnimationFrameCount - 1u);
            m_models.Renderer.LoopAnimation = m_loopAnimation;

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
