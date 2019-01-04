#include "scene_exporter.hpp"

namespace loader::exprt {

bool SceneExporter::save_scene()
{
	std::string jsonPath = m_filePath.string();
	jsonPath.append(".json");
	// TODO Open old Json

	rapidjson::Document document;
	rapidjson::Value objectToExport;
	objectToExport.SetObject();

	document.AddMember("version", FILE_VERSION, document.GetAllocator());
	std::string binaryPathString = m_filePath.string();
	binaryPathString.append(".mff");
	rapidjson::Value binaryPath;
	binaryPath.SetString(binaryPathString.c_str(), rapidjson::SizeType(binaryPathString.length()));
	document.AddMember("binary", binaryPath, document.GetAllocator());

	rapidjson::Value cameras;
	cameras.SetObject();
	size_t cameraCount = world_get_camera_count();

	for (size_t i = 0; i < cameraCount; i++)
	{
		rapidjson::Value camera;
		camera.SetObject();

		CameraHdl cameraHandle = world_get_camera_by_index(i);
		CameraType cameraType = world_get_camera_type(cameraHandle);

		switch (cameraType)
		{
		case CAM_PINHOLE:
			camera.AddMember("type", "pinhole", document.GetAllocator());
			float vFov;
			world_get_pinhole_camera_fov(cameraHandle, &vFov);
			camera.AddMember("fov", vFov, document.GetAllocator());
			break;
		case CAM_FOCUS:
			camera.AddMember("type", "focus", document.GetAllocator());
			float focalLength;
			world_get_focus_camera_focal_length(cameraHandle, &focalLength);
			camera.AddMember("focalLength", focalLength, document.GetAllocator());
			float chipHeight;
			world_get_focus_camera_sensor_height(cameraHandle, &chipHeight);
			camera.AddMember("chipHeight", chipHeight, document.GetAllocator());
			float focusDistance;
			world_get_focus_camera_focus_distance(cameraHandle, &focusDistance);
			camera.AddMember("focusDistance", focusDistance, document.GetAllocator());
			float aperture;
			world_get_focus_camera_aperture(cameraHandle, &aperture);
			camera.AddMember("focalLength", aperture, document.GetAllocator());
			break;
			// TODO ORTHO (not Implemented yet)
		default:
			// TODO Exception?
			break;
		}
	}

	return true;
}

} // namespace loader::exprt