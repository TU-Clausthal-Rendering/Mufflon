using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Effects;
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

            m_models.App.Window.BorderHost.SizeChanged += BorderHostOnSizeChanged;

            m_models.Viewport.PropertyChanged += ViewportOnPropertyChanged;
        }

        private void BorderHostOnSizeChanged(object sender, SizeChangedEventArgs args)
        {
            m_models.Viewport.Width = (int)args.NewSize.Width;
            m_models.Viewport.Height = (int)args.NewSize.Height;
        }

        private void ViewportOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(ViewportModel.RenderWidth):
                    OnPropertyChanged(nameof(ScrollMaximumX));
                    OnPropertyChanged(nameof(RenderWidth));
                    break;
                case nameof(ViewportModel.RenderHeight):
                    OnPropertyChanged(nameof(ScrollMaximumY));
                    OnPropertyChanged(nameof(RenderHeight));
                    break;
                case nameof(ViewportModel.Width):
                    OnPropertyChanged(nameof(ScrollWidth));
                    OnPropertyChanged(nameof(ScrollMaximumX));
                    break;
                case nameof(ViewportModel.Height):
                    OnPropertyChanged(nameof(ScrollHeight));
                    OnPropertyChanged(nameof(ScrollMaximumY));
                    break;
                case nameof(ViewportModel.OffsetX):
                    OnPropertyChanged(nameof(OffsetX));
                    break;
                case nameof(ViewportModel.OffsetY):
                    OnPropertyChanged(nameof(OffsetY));
                    break;
            }
        }

        // maximum size of the viewport
        public int RenderWidth => m_models.Viewport.RenderWidth;

        // maximum size if the viewport
        public int RenderHeight => m_models.Viewport.RenderHeight;

        // maximum for scroll bar
        public int ScrollMaximumX => m_models.Viewport.RenderWidth - ScrollWidth;

        // maximum for scroll bar
        public int ScrollMaximumY => m_models.Viewport.RenderHeight - ScrollHeight;

        // width of the scroll bars bar
        public int ScrollWidth => m_models.Viewport.Width;
        
        // height of the scroll bars bar
        public int ScrollHeight  => m_models.Viewport.Height;

        // scroll value
        public int OffsetX
        {
            get => m_models.Viewport.OffsetX;
            set => m_models.Viewport.OffsetX = value;
        }

        // scroll value
        public int OffsetY
        {
            get => m_models.Viewport.OffsetY;
            set => m_models.Viewport.OffsetY = value;
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
