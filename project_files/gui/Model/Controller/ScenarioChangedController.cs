using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace gui.Model.Controller
{
    /// <summary>
    /// sets properties outside of the world model in case the world scenario changed
    /// </summary>
    public class ScenarioChangedController
    {
        private readonly Models m_models;

        public ScenarioChangedController(Models models)
        {
            m_models = models;
            m_models.PropertyChanged += ModelsOnPropertyChanged;
        }

        private void ModelsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.World):
                    if (m_models.World != null)
                    {
                        AdjustScenarioViewport();
                    }
                    break;
            }
        }

        void AdjustScenarioViewport()
        {
            m_models.Viewport.RenderWidth = (int)m_models.World.CurrentScenario.Resolution.X;
            m_models.Viewport.RenderHeight = (int)m_models.World.CurrentScenario.Resolution.Y;
        }
    }
}
