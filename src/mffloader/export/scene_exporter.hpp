#pragma once

#include "util/filesystem.hpp"
#include <rapidjson/document.h>
#include "core/export/interface.h"

namespace mff_loader::exprt {

class SceneExporter
{
public:
	static constexpr const char FILE_VERSION[] = "1.0";

	SceneExporter(fs::path fileDestinationPath, fs::path mffPath) :
		m_fileDestinationPath(fs::canonical(fileDestinationPath)),
		m_mffPath(mffPath)
	{}

	bool save_scene() const;
private:
	bool save_cameras(rapidjson::Document& document) const;
	bool save_lights(rapidjson::Document& document) const;
	bool save_materials(rapidjson::Document& document) const;
	rapidjson::Value save_material( MaterialParams materialParams, rapidjson::Document& document ) const;
	bool save_scenarios(rapidjson::Document& document) const;

	bool add_member_from_texture_handle(const TextureHdl& textureHdl, std::string memberName, rapidjson::Value& saveIn, rapidjson::Document& document) const;

	rapidjson::Value store_in_array(Vec3 value, rapidjson::Document& document) const;
	rapidjson::Value store_in_array(Vec2 value, rapidjson::Document& document) const;
	template<typename T>
	rapidjson::Value store_in_array(std::vector<T> value, rapidjson::Document& document) const;
	rapidjson::Value store_in_array_from_float_string(std::string floatString, rapidjson::Document& document) const;

	rapidjson::Value store_in_string_relative_to_destination_path(const fs::path& path, rapidjson::Document& document) const;

	const fs::path m_fileDestinationPath;
	const fs::path m_mffPath;
};

/*class CutomPrettyWriter : public rapidjson::PrettyWriter<rapidjson::StringBuffer>
{
public:
	CutomPrettyWriter(rapidjson::StringBuffer& sBuf)
		: rapidjson::PrettyWriter<rapidjson::StringBuffer>(sBuf)
	{}
	bool Double(double d) { Prefix(rapidjson::kNumberType); return EndValue(WriteDouble(d)); }
	bool WriteDouble(double d) {
		if (rapidjson::internal::Double(d).IsNanOrInf()) {
			// Note: This code path can only be reached if (RAPIDJSON_WRITE_DEFAULT_FLAGS & kWriteNanAndInfFlag).
			if (!(rapidjson::kWriteDefaultFlags & rapidjson::kWriteNanAndInfFlag))
				return false;
			if (rapidjson::internal::Double(d).IsNan()) {
				PutReserve(*os_, 3);
				PutUnsafe(*os_, 'N'); PutUnsafe(*os_, 'a'); PutUnsafe(*os_, 'N');
				return true;
			}
			if (rapidjson::internal::Double(d).Sign()) {
				PutReserve(*os_, 9);
				PutUnsafe(*os_, '-');
			}
			else
				PutReserve(*os_, 8);
			PutUnsafe(*os_, 'I'); PutUnsafe(*os_, 'n'); PutUnsafe(*os_, 'f');
			PutUnsafe(*os_, 'i'); PutUnsafe(*os_, 'n'); PutUnsafe(*os_, 'i'); PutUnsafe(*os_, 't'); PutUnsafe(*os_, 'y');
			return true;
		}

		char *buffer = os_->Push(25);

		double dCopy = d;
		int i;
		if(d != 0.0)
		{
			for(i = 0; abs(dCopy) < 0.0; i++)
			{
				dCopy *= 10;
			}
		}
		char* end = rapidjson::internal::dtoa(d, buffer, 3 + i);
		os_->Pop(static_cast<size_t>(25 - (end - buffer)));
		return true;
	}
};*/


}// namespace mff_loader::exprt