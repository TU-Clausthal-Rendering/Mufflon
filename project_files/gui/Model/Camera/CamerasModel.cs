using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using gui.Dll;
using gui.Utility;

namespace gui.Model.Camera
{
    public class CamerasModel : SynchronizedModelList<CameraModel>
    {
        public CamerasModel()
        {
            var numCams = Core.world_get_camera_count();
            for (var i = 0u; i < numCams; ++i)
            {
                var handle = Core.world_get_camera_by_index(i);
                if(handle == IntPtr.Zero)
                    throw new Exception(Core.core_get_dll_error());

                var type = Core.world_get_camera_type(handle);
                m_list.Add(CameraModel.MakeFromHandle(handle, type));
            }
        }
    }
}
