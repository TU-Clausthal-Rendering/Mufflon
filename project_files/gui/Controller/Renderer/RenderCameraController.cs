using gui.Dll;
using gui.Model;
using gui.Utility;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;

namespace gui.Controller.Renderer
{
    public class RendererCameraController
    {
        private Models m_models;
        private UIElement m_referenceElement;
        private BitArray m_keyWasPressed = new BitArray(Enum.GetNames(typeof(Key)).Length);
        private Point m_lastMousePosition;

        public RendererCameraController(Models models, UIElement referenceElement) {
            m_models = models;
            m_referenceElement = referenceElement;
            referenceElement.KeyDown += OnKeyDownHandler;
            referenceElement.KeyUp += OnKeyUpHandler;
            referenceElement.MouseDown += OnMouseDownHandler;
            referenceElement.MouseMove += OnMouseMoveHandler;
            referenceElement.MouseWheel += OnMouseWheel;
            m_models.RendererCamera.OnMove += OnCameraMove;
        }

        public Vec2<float> MouseDrag { get; private set; }

        private void OnKeyDownHandler(object sender, KeyEventArgs e)
        {
            m_keyWasPressed[(int)e.Key] = true;

            var oldOffset = m_models.RendererCamera.KeyboardOffset;
            switch (e.Key)
            {
                case Key.W: oldOffset.Z = m_models.RendererCamera.KeyboardSpeed; break;
                case Key.S: oldOffset.Z = -m_models.RendererCamera.KeyboardSpeed; break;
                case Key.A: oldOffset.X = -m_models.RendererCamera.KeyboardSpeed; break;
                case Key.D: oldOffset.X = m_models.RendererCamera.KeyboardSpeed; break;
                case Key.E: oldOffset.Y = m_models.RendererCamera.KeyboardSpeed; break;
                case Key.Q: oldOffset.Y = -m_models.RendererCamera.KeyboardSpeed; break;
            }
            m_models.RendererCamera.KeyboardOffset = oldOffset;
        }

        private void OnKeyUpHandler(object sender, KeyEventArgs e)
        {
            m_keyWasPressed[(int)e.Key] = false;
        }

        private void OnMouseDownHandler(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                m_referenceElement.Focus();
                m_lastMousePosition = e.MouseDevice.GetPosition(m_referenceElement);
            }
        }

        private void OnMouseMoveHandler(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                Point currPos = e.MouseDevice.GetPosition(m_referenceElement);
                m_models.RendererCamera.MouseOffset += currPos - m_lastMousePosition;
                m_lastMousePosition = currPos;
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
            if (m_keyWasPressed[(int)Key.W]) oldOffset.Z = m_models.RendererCamera.KeyboardSpeed;
            if (m_keyWasPressed[(int)Key.S]) oldOffset.Z = -m_models.RendererCamera.KeyboardSpeed;
            if (m_keyWasPressed[(int)Key.A]) oldOffset.X = -m_models.RendererCamera.KeyboardSpeed;
            if (m_keyWasPressed[(int)Key.D]) oldOffset.X = m_models.RendererCamera.KeyboardSpeed;
            if (m_keyWasPressed[(int)Key.E]) oldOffset.Y = m_models.RendererCamera.KeyboardSpeed;
            if (m_keyWasPressed[(int)Key.Q]) oldOffset.Y = -m_models.RendererCamera.KeyboardSpeed;

            // Check if the camera moved
            if (oldOffset.X != 0f || oldOffset.Y != 0f || oldOffset.Z != 0f)
                m_models.World.CurrentScenario.Camera.Move(oldOffset);

            // Check if the camera turned
            if (m_models.RendererCamera.MouseOffset.X != 0f || m_models.RendererCamera.MouseOffset.Y != 0f)
            {
                Vec2<float> mouseOffset = new Vec2<float>(-(float)m_models.RendererCamera.MouseOffset.Y * m_models.RendererCamera.MouseSpeed,
                                                          -(float)m_models.RendererCamera.MouseOffset.X * m_models.RendererCamera.MouseSpeed);
                m_models.World.CurrentScenario.Camera.Rotate(mouseOffset);
            }
        }
    }
}
