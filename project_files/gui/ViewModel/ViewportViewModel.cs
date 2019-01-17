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

            int oldMaxOffsetX = DesiredWidth - Math.Min(m_models.Viewport.Width, DesiredWidth);
            int oldMaxOffsetY = DesiredHeight - Math.Min(m_models.Viewport.Height, DesiredHeight);

            // Recompute the offsets for the scrollbars
            m_models.Viewport.Width = (int)args.NewSize.Width;
            m_models.Viewport.Height = (int)args.NewSize.Height;
            int newMaxOffsetX = DesiredWidth - Math.Min(m_models.Viewport.Width, DesiredWidth);
            int newMaxOffsetY = DesiredHeight - Math.Min(m_models.Viewport.Height, DesiredHeight);
            // Adjust the offset so that it stays roughly the same fractionally
            // Do not change the offset if we have a zero-sized border
            if(oldMaxOffsetX != 0)
                OffsetX = (int)(OffsetX * newMaxOffsetX / (float)oldMaxOffsetX);
            if (oldMaxOffsetY != 0)
                OffsetY = (int)(OffsetY * newMaxOffsetY / (float)oldMaxOffsetY);

        }

        private void ViewportOnPropertyChanged(object sender, PropertyChangedEventArgs args)
        {
            switch (args.PropertyName)
            {
                case nameof(ViewportModel.RenderWidth):
                    OnPropertyChanged(nameof(RenderWidth));
                    OnPropertyChanged(nameof(DesiredWidth));
                    OnPropertyChanged(nameof(ScrollMaximumX));
                    break;
                case nameof(ViewportModel.RenderHeight):
                    OnPropertyChanged(nameof(RenderHeight));
                    OnPropertyChanged(nameof(DesiredHeight));
                    OnPropertyChanged(nameof(ScrollMaximumY));
                    break;
                case nameof(ViewportModel.DesiredWidth):
                    OnPropertyChanged(nameof(ScrollMaximumX));
                    OnPropertyChanged(nameof(DesiredWidth));
                    break;
                case nameof(ViewportModel.DesiredHeight):
                    OnPropertyChanged(nameof(ScrollMaximumY));
                    OnPropertyChanged(nameof(DesiredHeight));
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
                case nameof(ViewportModel.Zoom):
                    OnPropertyChanged(nameof(Zoom));
                    break;
            }
        }

        // maximum size of the viewport
        public int RenderWidth => m_models.Viewport.RenderWidth;

        // maximum size if the viewport
        public int RenderHeight => m_models.Viewport.RenderHeight;

        // effective maximum size including zoom
        public int DesiredWidth => m_models.Viewport.DesiredWidth;

        // effective maximum size including zoom
        public int DesiredHeight => m_models.Viewport.DesiredHeight;

        // maximum for scroll bar
        public int ScrollMaximumX => DesiredWidth - ScrollWidth;

        // maximum for scroll bar
        public int ScrollMaximumY => DesiredHeight - ScrollHeight;

        // width of the scroll bars bar
        public int ScrollWidth => m_models.Viewport.Width;
        
        // height of the scroll bars bar
        public int ScrollHeight  => m_models.Viewport.Height;

        public float Zoom => m_models.Viewport.Zoom;

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
