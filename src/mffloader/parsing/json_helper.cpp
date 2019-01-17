#include "json_helper.hpp"
#include <cstdint>

namespace mff_loader::json {

namespace {

std::string_view map_level_to_string(ParserState::Level lvl) {
	switch(lvl) {
		case ParserState::Level::CAMERAS: return "cameras";
		case ParserState::Level::LIGHTS: return "lights";
		case ParserState::Level::MATERIALS: return "materials";
		case ParserState::Level::ROOT: return "root";
		case ParserState::Level::SCENARIOS: return "scenarios";
		default: return "unknown";
	}
}

std::string_view map_type_to_string(ParserState::Value val) {
	switch(val) {
		case ParserState::Value::ARRAY: return "array";
		case ParserState::Value::BOOLEAN: return "boolean";
		case ParserState::Value::NULLVAL: return "null";
		case ParserState::Value::NUMBER: return "number";
		case ParserState::Value::OBJECT: return "object";
		case ParserState::Value::STRING: return "string";
		default: return "unknown";
	}
}

} // namespace

std::string ParserState::get_parser_level() const {
	std::string key;
	if(current == ParserState::Level::ROOT) {
		if(objectNames.empty())
			key = "\"Root\"";
		else
			key = "\"Root\":\"" + std::string(objectNames.front()) + "\":";
	} else {
		key = '\"' + std::string(map_level_to_string(current)) + "\":";
		for(std::size_t i = 0u; i < objectNames.size(); ++i)
			key += '\"' + std::string(objectNames[i]) + "\":";
		if(expected == ParserState::Value::NONE)
			key = key.substr(0u, key.length() - 1u);
		else
			key += '\"' + std::string(objectNames.back()) + '\"';
	}
	return key;
}

ParserException::ParserException(const ParserState& state) {
	// TODO
	if(state.current == ParserState::Level::ROOT) {
		if(state.objectNames.empty())
			m_msg = "Document root is not an object";
		else
			m_msg = "Root document is missing key '"
					+ std::string(state.objectNames.front()) + "'";
	} else {
		m_msg = "Key \"" + std::string(map_level_to_string(state.current)) + "\":";
		for(std::size_t i = 1u; i < state.objectNames.size(); ++i)
			m_msg += '\"' + std::string(state.objectNames[i - 1u]) + "\":";
		if(state.expected == ParserState::Value::NONE) {
			// Replace the last ':'
			m_msg[m_msg.length() - 1u] = ' ';
			m_msg += "is missing key \"" + std::string(state.objectNames.back()) + '\"';
		} else {
			m_msg += '\"' + std::string(state.objectNames.back()) + "\" has invalid type (expected "
				+ std::string(map_type_to_string(state.expected)) + ")";
		}
	}
}

rapidjson::Value::ConstMemberIterator get(ParserState& state,
										  const rapidjson::Value& val,
										  const char* name,
										  bool required) {
	rapidjson::Value::ConstMemberIterator iter = val.FindMember(name);
	if(required && iter == val.MemberEnd()) {
		state.objectNames.push_back(name);
		state.expected = ParserState::Value::NONE;
		throw ParserException(state);
	}
	return iter;
}

void assertObject(ParserState& state, const rapidjson::Value& val) {
	if(!val.IsObject()) {
		state.expected = ParserState::Value::OBJECT;
		throw ParserException(state);
	}
}
void assertObject(ParserState& state, const rapidjson::Value::ConstMemberIterator& val) {
	if(!val->value.IsObject()) {
		state.expected = ParserState::Value::OBJECT;
		state.objectNames.push_back(val->name.GetString());
		throw ParserException(state);
	}
}

void assertArray(ParserState& state, const rapidjson::Value& val) {
	if(!val.IsArray()) {
		state.expected = ParserState::Value::ARRAY;
		state.expectedArraySize = 0u;
		throw ParserException(state);
	}
}
void assertArray(ParserState& state, const rapidjson::Value::ConstMemberIterator& val) {
	if(!val->value.IsArray()) {
		state.expected = ParserState::Value::ARRAY;
		state.expectedArraySize = 0u;
		state.objectNames.push_back(val->name.GetString());
		throw ParserException(state);
	}
}

void assertArray(ParserState& state, const rapidjson::Value& val, std::size_t expected) {
	if(!val.IsArray()) {
		state.expected = ParserState::Value::ARRAY;
		state.expectedArraySize = expected;
		throw ParserException(state);
	}
}
void assertArray(ParserState& state, const rapidjson::Value::ConstMemberIterator& val, std::size_t expected) {
	if(!val->value.IsArray()) {
		state.expected = ParserState::Value::ARRAY;
		state.expectedArraySize = expected;
		state.objectNames.push_back(val->name.GetString());
		throw ParserException(state);
	}
}

void assertNumber(ParserState& state, const rapidjson::Value& val) {
	if(!val.IsNumber()) {
		state.expected = ParserState::Value::NUMBER;
		throw ParserException(state);
	}
}
void assertNumber(ParserState& state, const rapidjson::Value::ConstMemberIterator& val) {
	if(!val->value.IsNumber()) {
		state.expected = ParserState::Value::NUMBER;
		state.objectNames.push_back(val->name.GetString());
		throw ParserException(state);
	}
}

void assertBoolean(ParserState& state, const rapidjson::Value& val) {
	if(!val.IsBool()) {
		state.expected = ParserState::Value::BOOLEAN;
		throw ParserException(state);
	}
}
void assertBoolean(ParserState& state, const rapidjson::Value::ConstMemberIterator& val) {
	if(!val->value.IsBool()) {
		state.expected = ParserState::Value::BOOLEAN;
		state.objectNames.push_back(val->name.GetString());
		throw ParserException(state);
	}
}

void assertString(ParserState& state, const rapidjson::Value& val) {
	if(!val.IsString()) {
		state.expected = ParserState::Value::STRING;
		throw ParserException(state);
	}
}
void assertString(ParserState& state, const rapidjson::Value::ConstMemberIterator& val) {
	if(!val->value.IsString()) {
		state.expected = ParserState::Value::STRING;
		state.objectNames.push_back(val->name.GetString());
		throw ParserException(state);
	}
}

template <>
std::int32_t read<std::int32_t>(ParserState& state, const rapidjson::Value& val) {
	assertNumber(state, val);
	return val.GetInt();
}

template <>
std::uint32_t read<std::uint32_t>(ParserState& state, const rapidjson::Value& val) {
	assertNumber(state, val);
	return val.GetUint();
}

template <>
std::int64_t read<std::int64_t>(ParserState& state, const rapidjson::Value& val) {
	assertNumber(state, val);
	return val.GetInt64();
}

template <>
std::uint64_t read<std::uint64_t>(ParserState& state, const rapidjson::Value& val) {
	assertNumber(state, val);
	return val.GetUint64();
}

template <>
float read<float>(ParserState& state, const rapidjson::Value& val) {
	assertNumber(state, val);
	return val.GetFloat();
}

template <>
double read<double>(ParserState& state, const rapidjson::Value& val) {
	assertNumber(state, val);
	return val.GetDouble();
}

template <>
bool read<bool>(ParserState& state, const rapidjson::Value& val) {
	assertBoolean(state, val);
	return val.GetBool();
}

template <>
const char* read<const char*>(ParserState& state, const rapidjson::Value& val) {
	assertString(state, val);
	return val.GetString();
}

template <>
ei::Vec2 read<ei::Vec2>(ParserState& state, const rapidjson::Value& val) {
	assertArray(state, val);
	assertArray(state, val, 2u);
	return ei::Vec2{
		read<float>(state, val[0u]),
		read<float>(state, val[1u])
	};
}

template <>
ei::Vec3 read<ei::Vec3>(ParserState& state, const rapidjson::Value& val) {
	assertArray(state, val);
	assertArray(state, val, 3u);
	return ei::Vec3{
		read<float>(state, val[0u]),
		read<float>(state, val[1u]),
		read<float>(state, val[2u])
	};
}

template <>
ei::Vec4 read<ei::Vec4>(ParserState& state, const rapidjson::Value& val) {
	assertArray(state, val);
	assertArray(state, val, 4u);
	return ei::Vec4{
		read<float>(state, val[0u]),
		read<float>(state, val[1u]),
		read<float>(state, val[2u]),
		read<float>(state, val[3u])
	};
}

template <>
ei::IVec2 read<ei::IVec2>(ParserState& state, const rapidjson::Value& val) {
	assertArray(state, val);
	assertArray(state, val, 2u);
	return ei::IVec2{
		read<std::int32_t>(state, val[0u]),
		read<std::int32_t>(state, val[1u])
	};
}

template <>
ei::IVec3 read<ei::IVec3>(ParserState& state, const rapidjson::Value& val) {
	assertArray(state, val);
	assertArray(state, val, 3u);
	return ei::IVec3{
		read<std::int32_t>(state, val[0u]),
		read<std::int32_t>(state, val[1u]),
		read<std::int32_t>(state, val[2u])
	};
}

template <>
ei::IVec4 read<ei::IVec4>(ParserState& state, const rapidjson::Value& val) {
	assertArray(state, val);
	assertArray(state, val, 4u);
	return ei::IVec4{
		read<std::int32_t>(state, val[0u]),
		read<std::int32_t>(state, val[1u]),
		read<std::int32_t>(state, val[2u]),
		read<std::int32_t>(state, val[3u])
	};
}

} // namespace mff_loader::json