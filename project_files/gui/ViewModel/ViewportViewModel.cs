using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using gui.Annotations;
using gui.Model;

namespace gui.ViewModel
{
    public class ViewportViewModel : INotifyPropertyChanged
    {
        private readonly Models m_models;

        public ViewportViewModel(Models models)
        {
            m_models = models;

            // initial width and height will be set by the callback as well
            m_models.App.Window.RendererScrollViewer.SizeChanged += RendererScrollViewerOnSizeChanged;

            m_models.Viewport.PropertyChanged += ViewportOnPropertyChanged;
        }

        private void RendererScrollViewerOnSizeChanged(object sender, SizeChangedEventArgs args)
        {
            m_models.Viewport.Width = (int)args.NewSize.Width;
            m_models.Viewport.Height = (int)args.NewSize.Height;
        }

        private void ViewportOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(ViewportModel.RenderWidth):
                    OnPropertyChanged(nameof(RenderWidth));
                    break;
                case nameof(ViewportModel.RenderHeight):
                    OnPropertyChanged(nameof(RenderHeight));
                    break;
                //case nameof(ViewportModel.Width):
                //    OnPropertyChanged(nameof(Width));
                //    break;
                //case nameof(ViewportModel.Height):
                //    OnPropertyChanged(nameof(Height));
                //    break;
            }
        }

        public int RenderWidth
        {
            get => m_models.Viewport.RenderWidth;
            set => m_models.Viewport.RenderWidth = value;
        }

        public int RenderHeight
        {
            get => m_models.Viewport.RenderHeight;
            set => m_models.Viewport.RenderHeight = value;
        }

        //public int Width
        //{
        //    get => m_models.Viewport.Width;
        //    set => m_models.Viewport.Width = value;
        //}
        //
        //public int Height
        //{
        //    get => m_models.Viewport.Height;
        //    set => m_models.Viewport.Height = value;
        //}

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
