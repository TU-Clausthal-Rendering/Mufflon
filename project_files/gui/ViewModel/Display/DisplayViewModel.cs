using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using gui.Annotations;
using gui.Command;
using gui.Dll;
using gui.Model;
using gui.Model.Display;
using gui.Utility;

namespace gui.ViewModel.Display
{
    public class DisplayViewModel : INotifyPropertyChanged
    {
        private readonly Models m_models;
        private readonly FrameworkElement m_referenceElement;
        // Dictionary mapping commands related to post-processing to their keyboard shortcuts
        // These may not be modified by settings and are only available when the reference element is focused
        private Dictionary<Key, ICommand> m_keybindings = new Dictionary<Key, ICommand>();

        public DisplayViewModel(Models models) {
            m_models = models;
            // Mouse position etc. will be computed relative to the canvas (where one would expect it to be)
            m_referenceElement = models.App.Window.RenderDisplay.RenderCanvas;

            AdjustGammaUp = new AdjustGammaUpCommand(m_models);
            AdjustGammaDown = new AdjustGammaDownCommand(m_models);

            // Add keybindings
            m_keybindings.Add(Key.OemPlus, AdjustGammaUp);
            m_keybindings.Add(Key.Add, AdjustGammaUp);
            m_keybindings.Add(Key.OemMinus, AdjustGammaDown);
            m_keybindings.Add(Key.Subtract, AdjustGammaDown);

            models.App.Window.RenderDisplay.RenderDisplayScroller.ScrollChanged += OnScrollChanged;
            m_models.App.Window.MouseMove += OnMouseMove;
            m_referenceElement.MouseWheel += OnMouseWheel;
            models.App.Window.RenderDisplay.KeyDown += OnKeyDown;

            RenderImageSource = m_models.Display.RenderBitmap;
            m_models.Display.PropertyChanged += OnDisplayChanged;
            m_models.Display.Repainted += OnRepainted;
        }

        public ICommand AdjustGammaUp { get; }
        public ICommand AdjustGammaDown { get; }

        // Enables rendered image display and zooming of said image
        public ScaleTransform BitmapScaling { get; private set; } = new ScaleTransform(1.0, -1.0);
        public ImageSource RenderImageSource { get; private set; }

        // Pixel colors
        public string CursorPos { get; private set; } = "0, 0";

        public string PixelColorRed { get => m_models.Display.CurrentPixelColor.X.ToString("0.####"); }
        public string PixelColorGreen { get => m_models.Display.CurrentPixelColor.Y.ToString("0.####"); }
        public string PixelColorBlue { get => m_models.Display.CurrentPixelColor.Z.ToString("0.####"); }
        public string PixelColorAlpha { get => m_models.Display.CurrentPixelColor.W.ToString("0.####"); }

        public float GammaFactor { get => m_models.Display.GammaFactor; }

        private void OnKeyDown(object sender, KeyEventArgs args) {
            ICommand cmd;
            if (m_keybindings.TryGetValue(args.Key, out cmd) && cmd.CanExecute(null))
                cmd.Execute(null);
        }

        private void OnMouseWheel(object sender, MouseWheelEventArgs args) {
            if((Keyboard.Modifiers & ModifierKeys.Control) != 0) {
                float step = args.Delta < 0.0f ? 1.0f / 1.001f : 1.001f;
                float value = (float)Math.Pow(step, Math.Abs(args.Delta));
                m_models.Display.Zoom *= value;
            }
        }

        private void OnMouseMove(object sender, MouseEventArgs args) {
            var position = args.GetPosition(m_referenceElement);
            m_models.Display.CursorPos = new Vec2<int>((int)position.X, (int)position.Y);
        }

        private void OnDisplayChanged(object sender, PropertyChangedEventArgs args) {
            switch(args.PropertyName) {
                case nameof(DisplayModel.GammaFactor):
                    // We may directly repaint the display without going through the render thread
                    // because the factor gets multiplied in upon copy
                    m_models.Display.Repaint(m_models.RenderTargetSelection.VisibleTarget.Name,
                        m_models.RenderTargetSelection.IsVarianceVisible);
                    OnPropertyChanged(nameof(GammaFactor));
                    break;
                case nameof(DisplayModel.Zoom):
                    BitmapScaling = new ScaleTransform(m_models.Display.Zoom, -m_models.Display.Zoom);
                    OnPropertyChanged(nameof(BitmapScaling));
                    break;
                case nameof(DisplayModel.CursorPos):
                    CursorPos = m_models.Display.CursorPos.X.ToString() + ", " + m_models.Display.CursorPos.Y.ToString();
                    OnPropertyChanged(nameof(CursorPos));
                    break;
                case nameof(DisplayModel.CurrentPixelColor):
                    OnPropertyChanged(nameof(PixelColorRed));
                    OnPropertyChanged(nameof(PixelColorGreen));
                    OnPropertyChanged(nameof(PixelColorBlue));
                    OnPropertyChanged(nameof(PixelColorAlpha));
                    break;
            }
        }

        private void OnRepainted(object sender) {
            m_referenceElement.Dispatcher.BeginInvoke(new Action(() => {
                RenderImageSource = m_models.Display.RenderBitmap;
                OnPropertyChanged(nameof(RenderImageSource));
            }));
        }

        private void OnScrollChanged(object sender, ScrollChangedEventArgs args) {
            m_models.Display.ScrollOffset = new Vec2<int>((int)args.HorizontalOffset,
                (int)args.VerticalOffset);
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
