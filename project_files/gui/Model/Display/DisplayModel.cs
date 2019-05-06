using gui.Annotations;
using gui.Dll;
using gui.Utility;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Permissions;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace gui.Model.Display
{
    public class DisplayModel : INotifyPropertyChanged
    {
        private static readonly int INITIAL_WIDTH = 800;
        private static readonly int INITIAL_HEIGHT = 600;

        public event EventHandler Repainted;

        // Image to be displayed as render result
        public WriteableBitmap RenderBitmap { get; private set; } = new WriteableBitmap(INITIAL_WIDTH, INITIAL_HEIGHT, 96, 96,
            PixelFormats.Rgb24, null);

        // Zoom of the display
        private float m_zoom = 1f;
        public float Zoom
        {
            get => m_zoom;
            set
            {
                var clamped = Math.Min(Math.Max(value, 0.01f), 100.0f);
                if (clamped == m_zoom) return;
                m_zoom = clamped;
                OnPropertyChanged(nameof(Zoom));
            }
        }

        private float m_gammaFactor = 1f;
        public float GammaFactor
        {
            get => m_gammaFactor;
            set
            {
                if (value == m_gammaFactor) return;
                m_gammaFactor = value;
                OnPropertyChanged(nameof(GammaFactor));
            }
        }

        private Vec2<int> m_renderSize = new Vec2<int>(INITIAL_WIDTH, INITIAL_HEIGHT);
        public Vec2<int> RenderSize
        {
            get => m_renderSize;
            set
            {
                if (value == m_renderSize) return;
                m_renderSize = value;
                RenderBitmap = new WriteableBitmap(RenderSize.X, RenderSize.Y, 96, 96,
                    PixelFormats.Rgb24, null);
                OnPropertyChanged(nameof(RenderSize));
            }
        }

        public Vec2<int> ScrollOffset { get; set; } = new Vec2<int>(0, 0);

        // Cursor position relative to the display (pixel-aligned)
        private Vec2<int> m_cursorPos = new Vec2<int>(0, 0);
        public Vec2<int> CursorPos
        {
            get => m_cursorPos;
            set
            {
                if(m_cursorPos == value) return;
                m_cursorPos.X = Math.Min((int)((ScrollOffset.X + Math.Max(0, value.X)) / Zoom), RenderSize.X - 1);
                m_cursorPos.Y = Math.Min((int)((ScrollOffset.Y + Math.Max(0, value.Y)) / Zoom), RenderSize.Y - 1);

                float r, g, b, a;
                if (!Core.core_get_pixel_info((uint)m_cursorPos.X,
                    (uint)m_cursorPos.Y, true, out r, out g, out b, out a))
                    throw new Exception(Core.core_get_dll_error());
                CurrentPixelColor = new Vec4<float>(r, g, b, a);
                OnPropertyChanged(nameof(CursorPos));
            }
        }

        private Vec4<float> m_pixelColor;
        public Vec4<float> CurrentPixelColor {
            get => m_pixelColor;
            private set
            {
                if (value == m_pixelColor) return;
                m_pixelColor = value;
                OnPropertyChanged(nameof(CurrentPixelColor));
            }
        }

        public void Repaint(uint targetIndex, bool paintVariance) {
            var stopwatch = new System.Diagnostics.Stopwatch();
            stopwatch.Restart();
            // Renew the display texture (we don't really care)
            IntPtr targetPtr = IntPtr.Zero;
            if(!Core.core_get_target_image(targetIndex, paintVariance, OpenGlDisplay.TextureFormat.RGBA32F, true, out targetPtr))
                throw new Exception(Core.core_get_dll_error());

            System.Windows.Application.Current.Dispatcher.Invoke(new Action(() => {
                // Update the display bitmap
                RenderBitmap.Lock();
                if(!Core.core_copy_screen_texture_rgb(RenderBitmap.BackBuffer, GammaFactor))
                    throw new Exception(Core.core_get_dll_error());
                RenderBitmap.AddDirtyRect(new System.Windows.Int32Rect(0, 0, RenderSize.X, RenderSize.Y));
                RenderBitmap.Unlock();
            }));

            Repainted(this, null);
        }

        public void UpdateCurrentPixelColor() {
            float r, g, b, a;
            if (!OpenGlDisplay.opengldisplay_get_pixel_value((UInt32)CursorPos.X, (UInt32)CursorPos.Y,
                out r, out g, out b, out a))
                throw new Exception(OpenGlDisplay.opengldisplay_get_dll_error());
            CurrentPixelColor = new Vec4<float>(r, g, b, a);
        }

        // TODO: on pixel changed to update color


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
