using gui.Dll;
using gui.Model;
using gui.Utility;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;

namespace gui.Controller.Renderer
{
    public class RendererCameraController
    {
        // The keys we support for camera movement
        private enum ScanCodes : uint
        {
            Q = 0x10,
            W = 0x11,
            E = 0x12,
            A = 0x1E,
            S = 0x1F,
            D = 0x20
        }

        private Models m_models;
        private UIElement m_referenceElement;
        private BitArray m_keyWasPressed = new BitArray(Enum.GetNames(typeof(ScanCodes)).Length);
        private Point m_lastMousePosition;
        private bool m_repositioningCursor = false;

        public RendererCameraController(Models models, UIElement referenceElement) {
            m_models = models;
            m_referenceElement = referenceElement;
            referenceElement.KeyDown += OnKeyDownHandler;
            referenceElement.KeyUp += OnKeyUpHandler;
            referenceElement.MouseDown += OnMouseDownHandler;
            referenceElement.MouseMove += OnMouseMoveHandler;
            referenceElement.MouseWheel += OnMouseWheel;
            m_models.RendererCamera.OnMove += OnCameraMove;
            m_models.RendererCamera.PropertyChanged += OnFlightModeChanged;
        }

        public Vec2<float> MouseDrag { get; private set; }

        private void OnKeyDownHandler(object sender, KeyEventArgs e)
        {
            uint scancode = User32.MapVirtualKeyA((uint)KeyInterop.VirtualKeyFromKey(e.Key), User32.MapVirtualKeyMapTypes.MAPVK_VK_TO_VSC_EX);

            var oldOffset = m_models.RendererCamera.KeyboardOffset;
            switch(scancode) {
                case (uint)ScanCodes.W: oldOffset.Z = m_models.RendererCamera.KeyboardSpeed; m_keyWasPressed[1] = true; break;
                case (uint)ScanCodes.S: oldOffset.Z = -m_models.RendererCamera.KeyboardSpeed; m_keyWasPressed[4] = true; break;
                case (uint)ScanCodes.A: oldOffset.X = -m_models.RendererCamera.KeyboardSpeed; m_keyWasPressed[3] = true; break;
                case (uint)ScanCodes.D: oldOffset.X = m_models.RendererCamera.KeyboardSpeed; m_keyWasPressed[5] = true; break;
                case (uint)ScanCodes.E: oldOffset.Y = m_models.RendererCamera.KeyboardSpeed; m_keyWasPressed[2] = true; break;
                case (uint)ScanCodes.Q: oldOffset.Y = -m_models.RendererCamera.KeyboardSpeed; m_keyWasPressed[0] = true; break;
            }
            m_models.RendererCamera.KeyboardOffset = oldOffset;
        }

        private void OnKeyUpHandler(object sender, KeyEventArgs e)
        {
            uint scancode = User32.MapVirtualKeyA((uint)KeyInterop.VirtualKeyFromKey(e.Key), User32.MapVirtualKeyMapTypes.MAPVK_VK_TO_VSC_EX);
            switch (scancode) {
                case (uint)ScanCodes.W: m_keyWasPressed[1] = false; break;
                case (uint)ScanCodes.S: m_keyWasPressed[4] = false; break;
                case (uint)ScanCodes.A: m_keyWasPressed[3] = false; break;
                case (uint)ScanCodes.D: m_keyWasPressed[5] = false; break;
                case (uint)ScanCodes.E: m_keyWasPressed[2] = false; break;
                case (uint)ScanCodes.Q: m_keyWasPressed[0] = false; break;
            }
        }

        private void OnMouseDownHandler(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed) {
                m_referenceElement.Focus();
                if(m_models.RendererCamera.FreeFlightEnabled)
                    m_models.RendererCamera.FreeFlightEnabled = false;
                else
                    m_lastMousePosition = e.MouseDevice.GetPosition(m_referenceElement);
            }
        }

        private void OnMouseMoveHandler(object sender, MouseEventArgs e)
        {
            if (m_repositioningCursor)
            {
                // Since we set the mouse cursor manually to the center the event gets fired (delayed)
                // even though we don't want to move, but this gives us the perfect oportunity to update
                // the last position without whacky hacks
                m_repositioningCursor = false;
                m_lastMousePosition = e.MouseDevice.GetPosition(m_referenceElement);
                return;
            }

            if (e.LeftButton == MouseButtonState.Pressed || m_models.RendererCamera.FreeFlightEnabled) {
                Point currPos = e.MouseDevice.GetPosition(m_referenceElement);
                m_models.RendererCamera.MouseOffset += currPos - m_lastMousePosition;
                m_lastMousePosition = currPos;
                if (m_models.RendererCamera.FreeFlightEnabled)
                    SetCursorToRenderCanvasCenter();
            }
        }

        private void OnMouseWheel(object sender, MouseWheelEventArgs args)
        {
            if (Keyboard.Modifiers == ModifierKeys.None)
                m_models.RendererCamera.KeyboardSpeed = Math.Max(0f, m_models.RendererCamera.KeyboardSpeed + 0.000125f * args.Delta);
        }

        private void OnCameraMove(object sender, EventArgs args)
        {
            var oldOffset = m_models.RendererCamera.KeyboardOffset;
            // Check for leftover pressed keys
            if (m_keyWasPressed[1]) oldOffset.Z = m_models.RendererCamera.KeyboardSpeed;
            if (m_keyWasPressed[4]) oldOffset.Z = -m_models.RendererCamera.KeyboardSpeed;
            if (m_keyWasPressed[3]) oldOffset.X = -m_models.RendererCamera.KeyboardSpeed;
            if (m_keyWasPressed[5]) oldOffset.X = m_models.RendererCamera.KeyboardSpeed;
            if (m_keyWasPressed[2]) oldOffset.Y = m_models.RendererCamera.KeyboardSpeed;
            if (m_keyWasPressed[0]) oldOffset.Y = -m_models.RendererCamera.KeyboardSpeed;

            // Check if the camera moved
            if (oldOffset.X != 0f || oldOffset.Y != 0f || oldOffset.Z != 0f)
                m_models.World.CurrentScenario.Camera.Move(oldOffset);

            // Check if the camera turned
            if (m_models.RendererCamera.MouseOffset.X != 0f || m_models.RendererCamera.MouseOffset.Y != 0f) {
                float factor = 1f;
                if (m_models.Settings.InvertCameraControls && !m_models.RendererCamera.FreeFlightEnabled)
                    factor = -1f;
                Vec2<float> mouseOffset = new Vec2<float>(factor * -(float)m_models.RendererCamera.MouseOffset.Y * m_models.RendererCamera.MouseSpeed,
                                                          factor * (float)m_models.RendererCamera.MouseOffset.X * m_models.RendererCamera.MouseSpeed);
                m_models.World.CurrentScenario.Camera.Rotate(mouseOffset);
            }
        }

        private void OnFlightModeChanged(object sender, PropertyChangedEventArgs args)
        {
            if(args.PropertyName == nameof(Model.Display.RenderCameraModel.FreeFlightEnabled))
            {
                if (m_models.RendererCamera.FreeFlightEnabled)
                {
                    Mouse.OverrideCursor = Cursors.None;
                    SetCursorToRenderCanvasCenter();
                    // If we don't manually focus the canvas we won't be able to use the keyboard to move
                    m_referenceElement.Focus();
                }
                else
                {
                    Mouse.OverrideCursor = null;
                }
            }
        }

        private void SetCursorToRenderCanvasCenter()
        {
            // Tell our handler that we should ignore this event
            m_repositioningCursor = true;
            var localCenter = new Point(m_models.App.Window.RenderDisplay.RenderCanvas.ActualWidth / 2.0,
                m_models.App.Window.RenderDisplay.RenderCanvas.ActualHeight / 2.0);
            var screenCenter = m_models.App.Window.RenderDisplay.RenderCanvas.PointToScreen(localCenter);
            System.Windows.Forms.Cursor.Position = new System.Drawing.Point((int)screenCenter.X, (int)screenCenter.Y);
        }
    }
}
