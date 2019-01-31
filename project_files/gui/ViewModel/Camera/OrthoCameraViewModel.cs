using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Model;
using gui.Model.Camera;
using gui.View.Camera;

namespace gui.ViewModel.Camera
{
    public class OrthoCameraViewModel : CameraViewModel
    {
        private readonly OrthoCameraModel m_parent;

        public OrthoCameraViewModel(Models modelssss, OrthoCameraModel parent) : base(modelssss, parent)
        {
            m_parent = parent;
        }

        public override object CreateView()
        {
            return new CameraView(this, new OrthoCameraView());
        }

        public float Width
        {
            get => m_parent.Width;
            set => m_parent.Width = value;
        }

        public float Height
        {
            get => m_parent.Height;
            set => m_parent.Height = value;
        }
    }
}
