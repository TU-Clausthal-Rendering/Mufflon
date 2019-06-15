using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Model;

namespace gui.Command
{
    public class EnterFreeFlightMode : ICommand
    {
        private Models m_models;

        public EnterFreeFlightMode(Models models)
        {
            m_models = models;
            models.Renderer.PropertyChanged += RendererOnPropertyChanged;
            models.RendererCamera.PropertyChanged += RenderCameraOnPropertyChanged;
        }

        private void RendererOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            if (sender == m_models.Renderer && args.PropertyName == nameof(Models.Renderer.IsRendering))
                OnCanExecuteChanged();
        }

        private void RenderCameraOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            if (sender == m_models.RendererCamera && args.PropertyName == nameof(Model.Display.RenderCameraModel.FreeFlightEnabled))
                OnCanExecuteChanged();
        }

        public bool CanExecute(object parameter)
        {
            return !m_models.RendererCamera.FreeFlightEnabled &&
                m_models.Renderer.IsRendering;
        }

        public void Execute(object parameter)
        {
            m_models.RendererCamera.FreeFlightEnabled = true;
        }

        public event EventHandler CanExecuteChanged;

        protected virtual void OnCanExecuteChanged()
        {
            CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        }
    }
}
