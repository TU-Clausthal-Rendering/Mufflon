﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using gui.Model;

namespace gui.Command
{
    public class PlayPauseCommand : ICommand
    {
        private Models m_models;

        public PlayPauseCommand(Models models)
        {
            m_models = models;
        }

        public bool CanExecute(object parameter)
        {
            return m_models.Scene.IsLoaded;
        }

        public void Execute(object parameter)
        {
            m_models.Renderer.IsRendering = !m_models.Renderer.IsRendering;
        }

        public event EventHandler CanExecuteChanged
        {
            add
            {
                CommandManager.RequerySuggested += value;
            }
            remove
            {
                CommandManager.RequerySuggested -= value;
            }
        }
    }
}
