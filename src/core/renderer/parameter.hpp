#pragma once

#include "util/assert.hpp"
#include "util/int_types.hpp"
#include "util/log.hpp"
#include "util/string_view.hpp"
#include "core/memory/dyntype_memory.hpp"
#include <map>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace mufflon { namespace renderer {

enum class ParameterTypes {
	INT,
	FLOAT,
	BOOL,
	ENUM
};

struct ParamDesc {
	const char* name;
	ParameterTypes type;
};

class IParameterHandler {
public:
	virtual int get_num_parameters() const noexcept = 0;
	virtual ParamDesc get_param_desc(int idx) const noexcept = 0;

	void set_param(StringView name, int value) {
		if(void* ref = const_cast<void*>(get(name, ParameterTypes::INT, "int", "set")))
			*as<int>(ref) = value;
	}

	void set_param(StringView name, float value) {
		if(void* ref = const_cast<void*>(get(name, ParameterTypes::FLOAT, "float", "set")))
			*as<float>(ref) = value;
	}

	void set_param(StringView name, bool value) {
		if(void* ref = const_cast<void*>(get(name, ParameterTypes::BOOL, "bool", "set")))
			*as<bool>(ref) = value;
	}
	void set_param_enum(StringView name, int value) {
		if(void* ref = const_cast<void*>(get(name, ParameterTypes::ENUM, "enum", "set")))
			*as<int>(ref) = value;
	}

	int get_param_int(StringView name) const {
		if(const void* ref = get(name, ParameterTypes::INT, "int", "get"))
			return *as<int>(ref);
		return 0;
	}
	float get_param_float(StringView name) const {
		if(const void* ref = get(name, ParameterTypes::FLOAT, "float", "get"))
			return *as<float>(ref);
		return 0.0f;
	}
	bool get_param_bool(StringView name) const {
		if(const void* ref = get(name, ParameterTypes::BOOL, "bool", "get"))
			return *as<bool>(ref);
		return false;
	}
	int get_param_enum(StringView name) const {
		if(const void* ref = get(name, ParameterTypes::ENUM, "enum", "get"))
			return *as<int>(ref);
		return 0;
	}

	virtual u32 get_enum_value_count(StringView param) const noexcept = 0;
	virtual int get_enum_value(StringView param, const u32 index) const noexcept = 0;
	virtual const std::string& get_enum_value_name(StringView param, int name) const noexcept = 0;
	virtual int get_enum_name_value(StringView param, const std::string& name) const noexcept = 0;
	virtual u32 get_enum_index(StringView param, const int value) const noexcept = 0;

private:
	virtual const void* get(StringView name, ParameterTypes expected, const char* expectedName, const char* action) const = 0;
};

/*
 * Parameter manager class. By multiple inheritance this class automatically
 * generates generic enumerate and set methods which can be called renderer 
 * independent.
 * A renderer should inherit from this by enumerating all the parameter types:
 * class Example : public ParameterHandler<PMinPathLength, PMaxPathLength> {...};
 */
template< typename... Params >
class ParameterHandler : public IParameterHandler, public Params... {
public:
	ParameterHandler() {
		init<0, Params...>();
	}

	// Methods to enumerate parameters
	int get_num_parameters() const noexcept final { return sizeof...(Params); }
	ParamDesc get_param_desc(int idx) const noexcept final {
		mAssert(idx >= 0 && idx < get_num_parameters());
		return m_idxToParam[idx];
	}

	u32 get_enum_value_count(StringView param) const noexcept override {
		auto it = m_paramMap.find(param);
		if(it == m_paramMap.end()) {
			logWarning("[ParameterHandler::get_enum_value_count] Trying to get enum value count of an unknown parameter '", param, "'.");
			return 0u;
		}
		if(it->second.type != ParameterTypes::ENUM) {
			logError("[ParameterHandler::get_enum_value_count] Parameter '", param, "' is not of type 'enum'.");
			return 0u;
		}
		return it->second.enumMaps.valueToName != nullptr ? static_cast<u32>(it->second.enumMaps.valueToName->size()) : 0u;
	}

	const std::string& get_enum_value_name(StringView param, int value) const noexcept override {
		static std::string s_empty;
		auto it = m_paramMap.find(param);
		if(it == m_paramMap.end()) {
			logWarning("[ParameterHandler::get_enum_value_name] Trying to get enum value name of an unknown parameter '", param, "'.");
			return s_empty;
		}
		if(it->second.type != ParameterTypes::ENUM) {
			logError("[ParameterHandler::get_enum_value_name] Parameter '", param, "' is not of type 'enum'.");
			return s_empty;
		}
		const auto nameIter = it->second.enumMaps.valueToName->find(value);
		if(nameIter == it->second.enumMaps.valueToName->cend()) {
			logWarning("[ParameterHandler::get_enum_value_name] Trying to get enum value name of unknown value '", value, "'.");
			return s_empty;
		} else {
			return nameIter->second;
		}
	}

	int get_enum_name_value(StringView param, const std::string& name) const noexcept override {
		auto it = m_paramMap.find(param);
		if(it == m_paramMap.end()) {
			logWarning("[ParameterHandler::get_enum_name_value] Trying to get enum value name of an unknown parameter '", param, "'.");
			return 0;
		}
		if(it->second.type != ParameterTypes::ENUM) {
			logError("[ParameterHandler::get_enum_name_value] Parameter '", param, "' is not of type 'enum'.");
			return 0;
		}
		const auto valIter = it->second.enumMaps.nameToValue->find(name);
		if(valIter == it->second.enumMaps.nameToValue->cend()) {
			logWarning("[ParameterHandler::get_enum_name_value] Trying to get enum value of unknown name '", name, "'.");
			return 0;
		} else {
			return valIter->second;
		}
	}

	int get_enum_value(StringView param, const u32 index) const noexcept override {
		auto it = m_paramMap.find(param);
		if(it == m_paramMap.end()) {
			logWarning("[ParameterHandler::get_enum_value] Trying to get enum value name of an unknown parameter '", param, "'.");
			return 0;
		}
		if(it->second.type != ParameterTypes::ENUM) {
			logError("[ParameterHandler::get_enum_value] Parameter '", param, "' is not of type 'enum'.");
			return 0;
		}
		if(index >= static_cast<u32>(it->second.enumMaps.valueToName->size())) {
			logError("[ParameterHandler::get_enum_value] Enum index ", index, " is out of bounds for parameter '", param, "'.");
			return 0;
		}
		return it->second.enumMaps.indexToValue[index];
	}

	u32 get_enum_index(StringView param, const int value) const noexcept override {
		auto it = m_paramMap.find(param);
		if(it == m_paramMap.end()) {
			logWarning("[ParameterHandler::get_enum_index] Trying to get enum value name of an unknown parameter '", param, "'.");
			return 0;
		}
		if(it->second.type != ParameterTypes::ENUM) {
			logError("[ParameterHandler::get_enum_index] Parameter '", param, "' is not of type 'enum'.");
			return 0;
		}
		// TODO: there certainly are more efficient ways...
		for(u32 i = 0u; i < static_cast<u32>(it->second.enumMaps.valueToName->size()); ++i) {
			if(it->second.enumMaps.indexToValue[i] == value)
				return i;
		}
		logError("[ParameterHandler::get_enum_index] Parameter '", param, "' does not have a value '", value, "'.");
		return 0u;
	}

private:
	ParamDesc m_idxToParam[sizeof...(Params)];
	struct ParamRef {
		int offset;			// Use an offset rather than a pointer (self-reference) to keep value semantics in ParameterHandler.
		ParameterTypes type;
		struct EnumRefs {
			const int* indexToValue;										// Maps index to value
			const std::unordered_map<int, std::string>* valueToName;		// Maps from enum value to name
			const std::map<std::string, int, std::less<>>* nameToValue;		// Maps from enum name to value
		} enumMaps;
	};
	std::map<std::string, ParamRef, std::less<>> m_paramMap;

	// Helper to create the runtime mapping (index and name) to parameters
	template<int Idx> void init() {} // Recursion end
	template<int Idx, typename P, typename... Tail>
	void init() {
		m_idxToParam[Idx] = P::get_desc();
		const int offset = int(reinterpret_cast<char*>(static_cast<P*>(this)) - reinterpret_cast<char*>(this));
		const auto enumMaps = get_enum_map_ptrs<P>();
		auto iter = m_paramMap.emplace(m_idxToParam[Idx].name, ParamRef{offset, m_idxToParam[Idx].type,
									   enumMaps });
		init<Idx+1, Tail...>();
	}

	// Checks for presence of enum
	template < typename P, typename R = void >
	struct enable_if_type { using type = R; };
	template < typename P, typename Enable = void >
	struct HasEnum : std::false_type {};
	template < typename P >
	struct HasEnum<P, typename enable_if_type<typename P::Values>::type> : std::true_type {};

	// Returns pointers to enum mappings
	template < typename P >
	static std::enable_if_t<HasEnum<P>::value, typename ParamRef::EnumRefs> get_enum_map_ptrs() noexcept {
		static_assert(P::get_desc().type == ParameterTypes::ENUM,
					  "Parameter is not declared as type 'ENUM', but has enum 'Values'");

		const auto mapPair = P::get_name_map();
		return { std::get<0>(mapPair).data(), &std::get<1>(mapPair), &std::get<2>(mapPair) };
	}
	template < typename P >
	static std::enable_if_t<!HasEnum<P>::value, typename ParamRef::EnumRefs> get_enum_map_ptrs() noexcept {
		static_assert(P::get_desc().type != ParameterTypes::ENUM,
					  "Parameter is declared as type 'ENUM', but does not have enum 'Values' as required");
		return { nullptr, nullptr, nullptr };
	}

	// Helper to shorten the implementation of setters and getters
	const void* get(StringView name, ParameterTypes expected, const char* expectedName, const char* action) const final {
		auto it = m_paramMap.find(name);
		if(it == m_paramMap.end()) {
			logWarning("[ParameterHandler::", action, "_param] Trying to ", action, " an unknown parameter '", name, "'.");
			return nullptr;
		}
		if(it->second.type != expected) {
			logError("[ParameterHandler::", action, "_param] Parameter '", name, "' is not of type ", expectedName, ".");
			return nullptr;
		}
		return as<char>(this) + it->second.offset;
	}
};


// *** Specific Parameter Implementations ***

namespace parameter_details {

// Splits a string by the separator
inline std::vector<std::string> split_string(const std::string& str, const char sep = ',') {
	std::vector<std::string> vecString;
	std::string item;

	std::stringstream stringStream(str);
	while(std::getline(stringStream, item, sep))
		vecString.push_back(item);

	return vecString;
}

// This is taken largely from https://stackoverflow.com/questions/28828957/enum-to-string-in-modern-c11-c14-c17-and-future-c20/48820063
template < class T >
inline std::tuple<std::vector<int>, std::unordered_map<T, std::string>,
				  std::map<std::string, T, std::less<>>> generate_name_map(std::string str) {
	str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
	str.erase(std::remove(str.begin(), str.end(), '('), str.end());

	std::vector<std::string> enumTokens(split_string(str));
	std::vector<int> indexMap;
	std::unordered_map<T, std::string> retMap;
	std::map<std::string, T, std::less<>> valMap;
	T inxMap;

	inxMap = 0;
	for(auto iter = enumTokens.begin(); iter != enumTokens.end(); ++iter) {
		// Token: [EnumName | EnumName=EnumValue]
		std::string enumName;
		if(iter->find('=') == std::string::npos) {
			enumName = *iter;
		} else {
			std::vector<std::string> enumNameValue(split_string(*iter, '='));
			enumName = enumNameValue[0];
			//inxMap = static_cast<T>(enumNameValue[1]);
			if(std::is_unsigned<T>::value) {
				inxMap = static_cast<T>(std::stoull(enumNameValue[1], 0, 0));
			} else {
				inxMap = static_cast<T>(std::stoll(enumNameValue[1], 0, 0));
			}
		}
		indexMap.push_back(inxMap);
		retMap[inxMap] = enumName;
		valMap[enumName] = inxMap;
		++inxMap;
	}

	return { std::move(indexMap), std::move(retMap), std::move(valMap) };
}

} // namespace parameter_details

// This is taken largely from https://stackoverflow.com/questions/28828957/enum-to-string-in-modern-c11-c14-c17-and-future-c20/48820063
// Usage: nameInit is the name of the struct member as well as (optional) initialization.
// After that specify all enum members and optionally their initializers
#define PARAM_ENUM(nameInit, ...)																						\
	enum class Values : int {																							\
		__VA_ARGS__																										\
	};																													\
	static std::tuple<const std::vector<int>&, const std::unordered_map<int, std::string>&,								\
					  const std::map<std::string, int, std::less<>>&> get_name_map() {									\
		static auto map	= parameter_details::generate_name_map<int>(#__VA_ARGS__);										\
		return map;																										\
	}																													\
	Values nameInit

struct PSeed {
	int seed { 0 };
	static constexpr ParamDesc get_desc() noexcept {
		return {"Seed", ParameterTypes::INT};
	}
};

// Only show paths with at least minPathLength segments
struct PMinPathLength {
	int minPathLength { 0 };
	static constexpr ParamDesc get_desc() noexcept {
		return {"Min. path length", ParameterTypes::INT};
	}
};

// Only show paths with at most maxPathLength segments
struct PMaxPathLength {
	int maxPathLength { 16 };
	static constexpr ParamDesc get_desc() noexcept {
		return {"Max. path length", ParameterTypes::INT};
	}
};

// Perform multiple next event estimates per vertex.
struct PNeeCount {
	int neeCount { 1 };
	static constexpr ParamDesc get_desc() noexcept {
		return {"NEE count", ParameterTypes::INT};
	}
};

// Use the slower but higher quality guide for NEE
struct PNeePositionGuide {
	bool neeUsePositionGuide { false };
	static constexpr ParamDesc get_desc() noexcept {
		return {"NEE position guide", ParameterTypes::BOOL};
	}
};

// Merge radius (with respect to scene size)
struct PMergeRadius {
	float mergeRadius { 0.001f };
	static constexpr ParamDesc get_desc() noexcept {
		return {"Relative merge radius", ParameterTypes::FLOAT};
	}
};

// Enable progressive rendering (e.g. shrinking merge radius)
struct PProgressive {
	bool progressive { false };
	static constexpr ParamDesc get_desc() noexcept {
		return {"Progressive", ParameterTypes::BOOL};
	}
};

struct PWireframeThickness {
	float thickness { 0.025f };
	static constexpr ParamDesc get_desc() noexcept {
		return { "Border thickness", ParameterTypes::FLOAT };
	}
};

struct PWireframeNormalize {
	bool normalize = false;
	static constexpr ParamDesc get_desc() noexcept {
		return { "Normalize thickness", ParameterTypes::BOOL };
	}
};

}} // namespace mufflon::renderer