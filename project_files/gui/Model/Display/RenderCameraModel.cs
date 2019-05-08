using gui.Utility;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace gui.Model.Display
{
    public class RenderCameraModel
    {
        public event EventHandler OnMove;

        public float KeyboardSpeed { get; set; } = 0.125f;
        public float MouseSpeed { get; set; } = 0.0025f;

        // Offset incurred by pressed keyboard keys
        public Vec3<float> KeyboardOffset { get; set; } = new Vec3<float>(0f);
        public Vector MouseOffset { get; set; } = new Vector(0, 0);

        // Triggers the OnMove event and resets all offsets
        public void MoveCamera()
        {
            OnMove(this, null);
            KeyboardOffset = new Vec3<float>(0f);
            MouseOffset = new Vector(0, 0);
        }
    }
}
