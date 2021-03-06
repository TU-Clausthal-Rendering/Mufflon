﻿using gui.Annotations;
using gui.Dll;
using gui.Utility;
using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace gui.Model.Display
{
    public class DisplayModel : INotifyPropertyChanged
    {
        private static readonly int INITIAL_WIDTH = 800;
        private static readonly int INITIAL_HEIGHT = 600;

        public delegate void RepaintEventHandler(object sender);
        public delegate void RequestRepaintEventHandler();
        public event RepaintEventHandler Repainted;
        public event RequestRepaintEventHandler RequestRepaint;

        // Image to be displayed as render result
        public WriteableBitmap RenderBitmap { get; private set; } = new WriteableBitmap(INITIAL_WIDTH, INITIAL_HEIGHT, 96, 96,
            PixelFormats.Rgb128Float, null);

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
                    PixelFormats.Rgb128Float, null);
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
                m_cursorPos.X = Math.Min(RenderSize.X - 1, Math.Max(0, value.X));
                m_cursorPos.Y = Math.Min(RenderSize.Y - 1, Math.Max(0, value.Y));

                float r, g, b, a;
                if (!Core.mufflon_get_pixel_info((uint)m_cursorPos.X,
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

        // To be called ONLY from the Render Thread!
        public void Repaint(bool onlyFromRenderIteration = false)
        {
            if(onlyFromRenderIteration)
            {
                System.Windows.Application.Current.Dispatcher.Invoke(new Action(() =>
                {
                    // Update the display bitmap
                    // BitmapSource, while it seems like it should be faster, actually isn't; I guess BitmapSource.Create
                    // actually copies the memory instead of just using it, so our copy is simply faster
                    RenderBitmap.Lock();
                    if (!Core.mufflon_copy_screen_texture_rgba32(RenderBitmap.BackBuffer, GammaFactor))
                        throw new Exception(Core.core_get_dll_error());
                    RenderBitmap.AddDirtyRect(new System.Windows.Int32Rect(0, 0, RenderSize.X, RenderSize.Y));
                    RenderBitmap.Unlock();
                }));
                Repainted(this);
            }
        }

        // Do NOT call this from the render thread!
        public void TriggerRepaint() {
            RequestRepaint();

            System.Windows.Application.Current.Dispatcher.Invoke(new Action(() => {
                // Update the display bitmap
                // BitmapSource, while it seems like it should be faster, actually isn't; I guess BitmapSource.Create
                // actually copies the memory instead of just using it, so our copy is simply faster
                RenderBitmap.Lock();
                if (!Core.mufflon_copy_screen_texture_rgba32(RenderBitmap.BackBuffer, GammaFactor))
                    throw new Exception(Core.core_get_dll_error());
                RenderBitmap.AddDirtyRect(new System.Windows.Int32Rect(0, 0, RenderSize.X, RenderSize.Y));
                RenderBitmap.Unlock();
            }));
            Repainted(this);
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
