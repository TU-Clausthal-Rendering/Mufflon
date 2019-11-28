#pragma once

#include <ei/vector.hpp>
#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <array>
#include <vector>
#include <stdexcept>

namespace mff_loader::json {

struct ParserState {
	enum class Level {
		ROOT,
		CAMERAS,
		LIGHTS,
		MATERIALS,
		SCENARIOS
	};

	enum class Value {
		NUMBER,
		STRING,
		BOOLEAN,
		ARRAY,
		OBJECT,
		NULLVAL,
		NONE
	};

	Level current;
	Value expected;
	std::size_t expectedArraySize;
	std::vector<const char*> objectNames;

	void reset() {
		current = Level::ROOT;
		expected = Value::NONE;
		expectedArraySize = 0u;
		objectNames.clear();
	}

	std::string get_parser_level() const;
};

class ParserException : public std::exception {
public:
	ParserException(const ParserState& state);

	virtual const char* what() const override {
		return m_msg.c_str();
	}

private:
	std::string m_msg;
};

// Attempts to find the given key in the value object; throws if required = true and
// the key wasn't found
rapidjson::Value::ConstMemberIterator get(ParserState& state,
										  const rapidjson::Value& val,
										  const char* name,
										  bool required = true);

// Reads a value without modifying the state
template < class T >
T read(ParserState& state, const rapidjson::Value& val);

// Reads a value and pushes/pops the key name onto the state's object stack
template < class T >
inline T read(ParserState& state, const rapidjson::Value::ConstMemberIterator& val) {
	state.objectNames.push_back(val->name.GetString());
	T res = read<T>(state, val->value);
	state.objectNames.pop_back();
	return res;
}

// Reads a value if present without modifying the state
template < class T >
inline T read_opt(ParserState& state, const rapidjson::Value& parent,
		   const char* name, const T& defVal) {
	auto obj = get(state, parent, name, false);
	if(obj != parent.MemberEnd())
		return read<T>(state, obj);
	return defVal;
}

// Reads an array of any size
template < class T >
inline void read(ParserState& state, const rapidjson::Value::ConstMemberIterator& val,
		  std::vector<T>& vals) {
	if(!val->value.IsArray()) {
		vals.push_back(read<T>(state, val->value));
		return;
	}
	state.objectNames.push_back(val->name.GetString());
	// We cover two cases: Array of array and only array (= single array)
	if(val->value.Size() == 0)
		return;
	if(val->value[0].IsArray()) {
		vals.reserve(vals.size() + val->value.Size());
		for(rapidjson::SizeType i = 0u; i < val->value.Size(); ++i)
			vals.push_back(read<T>(state, val->value[i]));
	} else {
		vals.push_back(read<T>(state, val));
	}
	state.objectNames.pop_back();
}

// Reads an array of the expected size
template < class T >
inline void read(ParserState& state, const rapidjson::Value::ConstMemberIterator& val,
		  std::vector<T>& vals, std::size_t expectedSize) {
	if(!val->value.IsArray()) {
		state.expected = ParserState::Value::ARRAY;
		state.expectedArraySize = 0u;
		throw ParserException(state);
	}
	if(!val->value.IsArray()) {
		state.expected = ParserState::Value::ARRAY;
		state.expectedArraySize = expectedSize;
		throw ParserException(state);
	}
	state.objectNames.push_back(val->name.GetString());
	vals.reserve(vals.size() + val->value.Size());
	for(rapidjson::SizeType i = 0u; i < val->value.Size(); ++i)
		vals.push_back(read<T>(state, val->value[i]));
	state.objectNames.pop_back();
}

// Throws if value isn't an object
void assertObject(ParserState& state, const rapidjson::Value& val);
void assertObject(ParserState& state, const rapidjson::Value::ConstMemberIterator& val);
void assertArray(ParserState& state, const rapidjson::Value& val);
void assertArray(ParserState& state, const rapidjson::Value::ConstMemberIterator& val);
void assertArray(ParserState& state, const rapidjson::Value& val, std::size_t expected);
void assertArray(ParserState& state, const rapidjson::Value::ConstMemberIterator& val, std::size_t expected);
void assertNumber(ParserState& state, const rapidjson::Value& val);
void assertNumber(ParserState& state, const rapidjson::Value::ConstMemberIterator& val);
void assertBoolean(ParserState& state, const rapidjson::Value& val);
void assertBoolean(ParserState& state, const rapidjson::Value::ConstMemberIterator& val);
void assertString(ParserState& state, const rapidjson::Value& val);
void assertString(ParserState& state, const rapidjson::Value::ConstMemberIterator& val);

} // namespace mff_loader::json