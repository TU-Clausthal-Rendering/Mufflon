using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Model;
using gui.Dll;
using gui.Model.Scene;

namespace gui.Command
{
    public class ResetCommand : ICommand
    {
        private readonly Models m_models;

        public ResetCommand(Models models)
        {
            m_models = models;
            m_models.PropertyChanged += ModelOnPropertyChanged;
            if(m_models.World != null)
                m_models.World.PropertyChanged += WorldOnPropertyChanged;
        }

        private void ModelOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.World):
                    if (m_models.World != null)
                    {
                        m_models.World.PropertyChanged += WorldOnPropertyChanged;
                    }
                    OnCanExecuteChanged();
                    break;
            }
        }

        private void WorldOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(WorldModel.IsSane):
                    OnCanExecuteChanged();
                    break;
            }
        }

        public bool CanExecute(object parameter)
        {
            return m_models.World != null && m_models.World.IsSane;
        }

        public void Execute(object parameter)
        {
            m_models.Renderer.Reset();
        }

        public event EventHandler CanExecuteChanged;

        protected virtual void OnCanExecuteChanged()
        {
            CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        }
    }
}
