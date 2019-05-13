using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using gui.Model;
using gui.Model.Events;
using gui.View;

namespace gui.ViewModel.Dialog
{
    public class LoadWorldViewModel
    {
        private readonly Models m_models;
        private SceneLoadStatus m_cancelDialog;
        private bool hasCancelled = false;
        private string loadingPath = "";

        public LoadWorldViewModel(Models models)
        {
            m_models = models;
            m_models.OnWorldLoad += ModelsOnOnWorldLoad;
        }

        private void ModelsOnOnWorldLoad(object sender, LoadEventArgs args)
        {
            switch (args.Status)
            {
                case LoadEventArgs.LoadStatus.Started:
                    loadingPath = args.Message;
                    m_cancelDialog = new SceneLoadStatus(Path.GetFileName(loadingPath));
                    m_cancelDialog.PropertyChanged += CancelDialogOnPropertyChanged;
                    hasCancelled = false;
                    break;
                case LoadEventArgs.LoadStatus.Loading:
                    break;
                case LoadEventArgs.LoadStatus.Failed:
                    m_cancelDialog?.Close();
                    m_cancelDialog = null;

                    if (!hasCancelled && m_models.Settings.LastWorlds.Contains(loadingPath))
                    {
                        if (MessageBox.Show("World file could not be loaded: " + args.Message + "; should it " +
                                            "be removed from the list of recent scenes?", "Unable to load scene", MessageBoxButton.YesNo,
                                MessageBoxImage.Error) == MessageBoxResult.Yes)
                        {
                            int index = m_models.Settings.LastWorlds.IndexOf(loadingPath);
                            if (index >= 0)
                            {
                                m_models.Settings.LastWorlds.RemoveAt(index);
                            }
                        }
                    }
                    else
                    {
                        MessageBox.Show(args.Message, "Unable to load scene", MessageBoxButton.OK);
                    }
                        
                    
                    break;
                case LoadEventArgs.LoadStatus.Finished:
                    m_cancelDialog?.Close();
                    m_cancelDialog = null;
                    break;
            }
        }

        private void CancelDialogOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(SceneLoadStatus.Canceled):
                    hasCancelled = true;
                    m_models.CancelSceneLoad();
                    break;
            }
        }
    }
}
