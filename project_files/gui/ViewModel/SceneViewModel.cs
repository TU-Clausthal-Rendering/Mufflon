using System;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using gui.Annotations;
using gui.Command;
using gui.Dll;
using gui.Model;
using gui.Model.Camera;
using gui.Model.Light;
using gui.Model.Material;
using gui.Model.Scene;
using gui.Properties;
using gui.Utility;
using gui.View;
using gui.View.Helper;

namespace gui.ViewModel
{
    public class SceneViewModel : INotifyPropertyChanged
    {
        public class SceneMenuItem : LoadSceneCommand
        {
            private Models m_models;

            public string Filename { get; set; }
            public string Path { get; set; }
            public ICommand Command { get; }

            public SceneMenuItem(Models models) : base(models)
            {
                m_models = models;
                Command = this;
            }

            public override void Execute(object parameter) => LoadScene(Path);
        }

        private readonly Models m_models;
        public ObservableCollection<SceneMenuItem> LastScenes { get; } = new ObservableCollection<SceneMenuItem>();

        private ComboBoxItem<ScenarioModel> m_selectedScenario = null;
        public ComboBoxItem<ScenarioModel> SelectedScenario
        {
            get => m_selectedScenario;
            set
            {
                if(ReferenceEquals(value, m_selectedScenario)) return;
                m_selectedScenario = value;
                OnPropertyChanged(nameof(SelectedScenario));

                // load selected scenario
                if (m_selectedScenario == null) return;
                m_models.World.CurrentScenario = m_selectedScenario.Cargo;
            }
        }
        public ObservableCollection<ComboBoxItem<ScenarioModel>> Scenarios { get; } = new ObservableCollection<ComboBoxItem<ScenarioModel>>();
        public bool CanLoadLastScenes => LastScenes.Count > 0 && !m_models.Renderer.IsRendering;

        public SceneViewModel(Models models)
        {
            m_models = models;

            foreach (string path in m_models.Settings.LastWorlds)
            {
                LastScenes.Add(new SceneMenuItem(m_models)
                {
                    Filename = Path.GetFileName(path),
                    Path = path
                });
            }

            m_models.Settings.LastWorlds.CollectionChanged += LastWorldsOnCollectionChanged;

            // assume no scene loaded
            Debug.Assert(m_models.World == null);
            m_models.PropertyChanged += ModelsOnPropertyChanged;
            m_models.Renderer.PropertyChanged += OnRendererChange;
        }

        private void LastWorldsOnCollectionChanged(object sender, NotifyCollectionChangedEventArgs args)
        {
            // TODO make more efficient by using args
            LastScenes.Clear();
            foreach (string path in m_models.Settings.LastWorlds)
            {
                LastScenes.Add(new SceneMenuItem(m_models)
                {
                    Filename = Path.GetFileName(path),
                    Path = path
                });
            }
            OnPropertyChanged(nameof(CanLoadLastScenes));
        }

        private void ModelsOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(Models.World):
                    // subscribe to new model
                    if (m_models.World != null)
                    {
                        m_models.World.Scenarios.CollectionChanged += ScenariosOnCollectionChanged;
                    }

                    // refresh views
                    MakeScenerioViews();

                    break;
            }
        }

        private void ScenariosOnCollectionChanged(object sender, NotifyCollectionChangedEventArgs e)
        {
            // TODO make more efficient for single value insertions
            MakeScenerioViews();
        }

        /// <summary>
        /// clears Scenarios collection and fills it with the new values.
        /// Sets SelectedScenario as well.
        /// </summary>
        private void MakeScenerioViews()
        {
            Scenarios.Clear();

            if (m_models.World == null)
            {
                SelectedScenario = null;
                return;
            }

            foreach (var scenario in m_models.World.Scenarios)
            {
                // add scenario view
                var view = new ComboBoxItem<ScenarioModel>(scenario.Name, scenario);
                Scenarios.Add(view);
                // set selected item
                if (ReferenceEquals(scenario, m_models.World.CurrentScenario))
                    SelectedScenario = view;
            }
        }

        private void OnRendererChange(object sender, PropertyChangedEventArgs args)
        {
            switch(args.PropertyName)
            {
                case nameof(RendererModel.IsRendering):
                    OnPropertyChanged(nameof(CanLoadLastScenes));
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
