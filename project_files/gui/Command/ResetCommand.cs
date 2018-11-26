using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Model;
using gui.Dll;

namespace gui.Command
{
    public class ResetCommand : ICommand
    {
        private Models m_models;

        public ResetCommand(Models models)
        {
            m_models = models;
            m_models.Renderer.PropertyChanged += RendererPropertyChanged;
        }

        private void RendererPropertyChanged(object sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            switch (e.PropertyName) {
                case nameof(RendererModel.IsRendering): OnCanExecuteChanged(); break;
            }
        }

        public bool CanExecute(object parameter)
        {
            return m_models.Renderer.IsRendering;
        }

        public void Execute(object parameter)
        {
            Core.render_reset();
        }

        public event EventHandler CanExecuteChanged;

        protected void OnCanExecuteChanged()
        {
            CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        }
    }
}
