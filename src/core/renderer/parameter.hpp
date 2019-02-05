#pragma once

#include "util/assert.hpp"
#include "util/log.hpp"
#include "util/string_view.hpp"
#include "core/memory/dyntype_memory.hpp"
#include <map>

namespace mufflon { namespace renderer {

enum class ParameterTypes {
	INT,
	FLOAT,
	BOOL
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
private:
	ParamDesc m_idxToParam[sizeof...(Params)];
	struct ParamRef {
		int offset;			// Use an offset rather than a pointer (self-reference) to keep value semantics in ParameterHandler.
		ParameterTypes type;
	};
	std::map<std::string, ParamRef, std::less<>> m_paramMap;

	// Helper to create the runtime mapping (index and name) to parameters
	template<int Idx> void init() {} // Recursion end
	template<int Idx, typename P, typename... Tail>
	void init() {
		m_idxToParam[Idx] = P::get_desc();
		int offset = int(reinterpret_cast<char*>(static_cast<P*>(this)) - reinterpret_cast<char*>(this));
		m_paramMap.emplace(m_idxToParam[Idx].name, ParamRef{offset, m_idxToParam[Idx].type});
		init<Idx+1, Tail...>();
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

// Only show paths with at least minPathLength segments
struct PMinPathLength {
	int minPathLength { 0 };
	static ParamDesc get_desc() noexcept {
		return {"Min. path length", ParameterTypes::INT};
	}
};

// Only show paths with at most maxPathLength segments
struct PMaxPathLength {
	int maxPathLength { 16 };
	static ParamDesc get_desc() noexcept {
		return {"Max. path length", ParameterTypes::INT};
	}
};

// Perform multiple next event estimates per vertex.
struct PNeeCount {
	int neeCount { 1 };
	static ParamDesc get_desc() noexcept {
		return {"NEE count", ParameterTypes::INT};
	}
};

// Use the slower but higher quality guide for NEE
struct PNeePositionGuide {
	bool neeUsePositionGuide { false };
	static ParamDesc get_desc() noexcept {
		return {"NEE position guide", ParameterTypes::BOOL};
	}
};

// Merge radius (with respect to scene size)
struct PMergeRadius {
	float mergeRadius { 0.001f };
	static ParamDesc get_desc() noexcept {
		return {"Relative merge radius", ParameterTypes::FLOAT};
	}
};

// Enable progressive rendering (e.g. shrinking merge radius)
struct PProgressive {
	bool progressive { false };
	static ParamDesc get_desc() noexcept {
		return {"Progressive", ParameterTypes::BOOL};
	}
};

struct PWireframeThickness {
	float thickness { 0.025f };
	static ParamDesc get_desc() noexcept {
		return { "Border thickness", ParameterTypes::FLOAT };
	}
};

struct PWireframeNormalize {
	bool normalize = false;
	static ParamDesc get_desc() noexcept {
		return { "Normalize thickness", ParameterTypes::BOOL };
	}
};

}} // namespace mufflon::renderer