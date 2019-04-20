using System;
using System.Collections;
using System.Windows;
using System.Windows.Input;
using gui.Model;
using gui.Utility;

namespace gui.Controller
{
    public class CameraController
    {
        private Models m_models;
        private UIElement m_referenceElement;
        private float m_keyboardSpeed = 0.125f;
        private float m_mouseSpeed = 0.0025f;
        private BitArray m_keyWasPressed = new BitArray(Enum.GetNames(typeof(Key)).Length);
        private Point m_lastMousePosition;

        private Vec3<float> m_keyboardOffset = new Vec3<float>(0f);
        private Vector m_mouseOffset = new Vector(0, 0);

        public CameraController(Models models, UIElement referenceElement)
        {
            m_models = models;
            m_referenceElement = referenceElement;
            referenceElement.KeyDown += OnKeyDownHandler;
            referenceElement.KeyUp += OnKeyUpHandler;
            referenceElement.MouseDown += OnMouseDownHandler;
            referenceElement.MouseMove += OnMouseMoveHandler;
            referenceElement.MouseWheel += OnMouseWheel;
            m_models.App.GlHost.UpdateCamera += UpdateCamera;
        }
        
        public Vec2<float> MouseDrag { get; private set; }

        private void OnKeyDownHandler(object sender, KeyEventArgs e)
        {
            m_keyWasPressed[(int)e.Key] = true;

            switch (e.Key)
            {
                case Key.W: m_keyboardOffset.Z = m_keyboardSpeed; break;
                case Key.A: m_keyboardOffset.Z = -m_keyboardSpeed; break;
                case Key.S: m_keyboardOffset.X = -m_keyboardSpeed; break;
                case Key.D: m_keyboardOffset.X = m_keyboardSpeed; break;
                case Key.E: m_keyboardOffset.Y = m_keyboardSpeed; break;
                case Key.Q: m_keyboardOffset.Y = -m_keyboardSpeed; break;
            }
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
                m_mouseOffset += currPos - m_lastMousePosition;
                m_lastMousePosition = currPos;
            }
        }

        private void OnMouseWheel(object sender, MouseWheelEventArgs args)
        {
            m_keyboardSpeed = Math.Max(0f, m_keyboardSpeed + 0.000125f * args.Delta);
        }

        private void UpdateCamera(object sender, EventArgs args)
        {
            // First check for leftover pressed keys
            if (m_keyWasPressed[(int)Key.W]) m_keyboardOffset.Z = m_keyboardSpeed;
            if (m_keyWasPressed[(int)Key.S]) m_keyboardOffset.Z = -m_keyboardSpeed;
            if (m_keyWasPressed[(int)Key.A]) m_keyboardOffset.X = -m_keyboardSpeed;
            if (m_keyWasPressed[(int)Key.D]) m_keyboardOffset.X = m_keyboardSpeed;
            if (m_keyWasPressed[(int)Key.E]) m_keyboardOffset.Y = m_keyboardSpeed;
            if (m_keyWasPressed[(int)Key.Q]) m_keyboardOffset.Y = -m_keyboardSpeed;

            // Check if the camera moved
            if(m_keyboardOffset.X != 0f || m_keyboardOffset.Y != 0f || m_keyboardOffset.Z != 0f)
            {
                m_models.World.CurrentScenario.Camera.Move(m_keyboardOffset);
                m_keyboardOffset = new Vec3<float>(0f);
            }

            // Check if the camera turned
            if (m_mouseOffset.X != 0f || m_mouseOffset.Y != 0f)
            {
                m_models.World.CurrentScenario.Camera.Rotate(new Vec2<float>(m_mouseSpeed * (float)m_mouseOffset.Y,
                                                                             -m_mouseSpeed * (float)m_mouseOffset.X));
                m_mouseOffset = new Vector(0, 0);
            }
        }
    }
}
