#include "profiling.hpp"
#include "util/filesystem.hpp"
#include "util/log.hpp"
#include <fstream>
#include <sstream>

namespace mufflon {

ProfileState::ProfileScope::ProfileScope(ProfileState& state) :
	m_state(&state) {}

ProfileState::ProfileScope::ProfileScope(ProfileScope&& scope) :
	m_state(scope.m_state) {
	scope.m_state = nullptr;
}

ProfileState::ProfileScope::~ProfileScope() {
	if(m_state != nullptr)
		m_state->add_sample();
}

ProfileState::ProfileState(ProfileState*& active) :
	m_activeRef(active),
	m_parent(active) {}

ProfileState* ProfileState::find_child(StringView name) {
	auto iter = m_children.find(name);
	if(iter != m_children.end())
		return iter->second.get();
	return nullptr;
}

ProfileState* ProfileState::add_child(StringView name, std::unique_ptr<ProfileState>&& child) {
	return m_children.emplace(name, std::move(child)).first->second.get();
}

void ProfileState::add_sample() {
	m_activeRef = m_parent;
	this->create_sample();
}

ProfileState::ProfileScope ProfileState::start() {
	m_activeRef = this;
	this->start_sample();
	return ProfileScope(*this);
}

void ProfileState::reset_all() {
	this->reset_sample();
	for(auto& child : m_children)
		child.second->reset_all();
}

void ProfileState::create_snapshot_all() {
	this->create_snapshot();
	for(auto& child : m_children)
		child.second->create_snapshot_all();
}

std::ostream& ProfileState::save_current_state(std::ostream& stream) const {
	stream << "children:" << m_children.size();
	this->save_profiler_current_state(stream);
	for(auto& child : m_children) {
		stream << '"' << child.first << "\",";
		child.second->save_current_state(stream);
	}
	return stream;
}

std::ostream& ProfileState::save_snapshots(std::ostream& stream) const {
	// Stores the snapshots as a CSV
	stream << "children:" << m_children.size();
	this->save_profiler_snapshots(stream);
	for(auto& child : m_children) {
		stream << '"' << child.first << "\",";
		child.second->save_snapshots(stream);
	}
	return stream;
}

std::ostream& ProfileState::save_total(std::ostream& stream) const {
	// Stores the snapshots as a CSV
	stream << "children:" << m_children.size();
	this->save_profiler_total(stream);
	for(auto& child : m_children) {
		stream << '"' << child.first << "\",";
		child.second->save_total(stream);
	}
	return stream;
}

std::ostream& ProfileState::save_total_and_snapshots(std::ostream& stream) const {
	// Stores the snapshots as a CSV
	stream << "children:" << m_children.size();
	this->save_profiler_total_and_snapshots(stream);
	for(auto& child : m_children) {
		stream << '"' << child.first << "\",";
		child.second->save_total_and_snapshots(stream);
	}
	return stream;
}

Profiler& Profiler::core() {
	static Profiler instance;
	return instance;
}

Profiler& Profiler::loader() {
	static Profiler instance;
	return instance;
}

void Profiler::reset_all() {
	for(auto& profiler : m_profilers)
		profiler.second->reset_all();
}

void Profiler::reset(StringView name) {
	if(m_active != nullptr) {
		// Use cascaded profiler
		ProfileState* state = m_active->find_child(name);
		if(state != nullptr)
			state->reset_sample();
	} else {
		// Use top level profiler
		auto iter = m_profilers.find(name);
		if(iter != m_profilers.end())
			iter->second->reset_sample();
	}
}

void Profiler::reset_from(StringView name) {
	if(m_active != nullptr) {
		// Use cascaded profiler
		ProfileState* state = m_active->find_child(name);
		if(state != nullptr)
			state->reset_all();
	} else {
		// Use top level profiler
		auto iter = m_profilers.find(name);
		if(iter != m_profilers.end())
			iter->second->reset_all();
	}
}

void Profiler::create_snapshot(StringView name) {
	if(m_active != nullptr) {
		// Use cascaded profiler
		ProfileState* state = m_active->find_child(name);
		if(state != nullptr)
			state->create_snapshot();
	} else {
		// Use top level profiler
		auto iter = m_profilers.find(name);
		if(iter != m_profilers.end())
			iter->second->create_snapshot();
	}
}

void Profiler::create_snapshot_from(StringView name) {
	if(m_active != nullptr) {
		// Use cascaded profiler
		ProfileState* state = m_active->find_child(name);
		if(state != nullptr)
			state->create_snapshot_all();
	} else {
		// Use top level profiler
		auto iter = m_profilers.find(name);
		if(iter != m_profilers.end())
			iter->second->create_snapshot_all();
	}
}

void Profiler::create_snapshot_all() {
	for(auto& profiler : m_profilers)
		profiler.second->create_snapshot_all();
}

void Profiler::save_current_state(StringView path) const {
	const auto file = fs::u8path(path.cbegin(), path.cend());
	std::ofstream fileStream(file);
	if(fileStream.bad()) {
		logError("[Profiler::save_current_state] could not open output file '",
				 file.string(), "'");
		return;
	}
	fileStream.exceptions(std::ios::failbit);

	try {
		for(auto& profiler : m_profilers) {
			fileStream << '"' << profiler.first << "\",";
			profiler.second->save_current_state(fileStream);
		}
	} catch(const std::exception& e) {
		logError("[Profiler::save_current_state] Failed to save profiling results: ",
				 e.what());
	}
}

void Profiler::save_snapshots(StringView path) const {
	const auto file = fs::u8path(path.cbegin(), path.cend());
	std::ofstream fileStream(file);
	if(fileStream.bad()) {
		logError("[Profiler::save_snapshots] could not open output file '",
				 file.string(), "'");
		return;
	}
	fileStream.exceptions(std::ios::failbit);

	try {
		for(auto& profiler : m_profilers) {
			fileStream << '"' << profiler.first << "\",";
			profiler.second->save_snapshots(fileStream);
		}
	} catch(const std::exception& e) {
		logError("[Profiler::save_snapshots] Failed to save profiling results: ",
				 e.what());
	}
}

void Profiler::save_total_and_snapshots(StringView path) const {
	const auto file = fs::u8path(path.cbegin(), path.cend());
	std::ofstream fileStream(file);
	if(fileStream.bad()) {
		logError("[Profiler::save_snapshots] could not open output file '",
				 file.string(), "'");
		return;
	}
	fileStream.exceptions(std::ios::failbit);

	try {
		for(auto& profiler : m_profilers) {
			fileStream << '"' << profiler.first << "\",";
			profiler.second->save_total_and_snapshots(fileStream);
		}
	} catch(const std::exception& e) {
		logError("[Profiler::save_snapshots] Failed to save profiling results: ",
				 e.what());
	}
}

std::string Profiler::save_current_state() const {
	std::ostringstream stream;
	try {
		for(auto& profiler : m_profilers) {
			stream << '"' << profiler.first << "\",";
			profiler.second->save_current_state(stream);
		}
	} catch(const std::exception& e) {
		logError("[Profiler::save_current_state] Failed to save profiling results: ",
				 e.what());
	}
	return stream.str();
}

std::string Profiler::save_snapshots() const {
	std::ostringstream stream;
	try {
		for(auto& profiler : m_profilers) {
			stream << '"' << profiler.first << "\",";
			profiler.second->save_snapshots(stream);
		}
	} catch(const std::exception& e) {
		logError("[Profiler::save_snapshots] Failed to save profiling results: ",
				 e.what());
	}
	return stream.str();
}

std::string Profiler::save_total() const {
	std::ostringstream stream;
	try {
		for(auto& profiler : m_profilers) {
			stream << '"' << profiler.first << "\",";
			profiler.second->save_total(stream);
		}
	} catch(const std::exception& e) {
		logError("[Profiler::save_snapshots] Failed to save profiling results: ",
				 e.what());
	}
	return stream.str();
}

std::string Profiler::save_total_and_snapshots() const {
	std::ostringstream stream;
	try {
		for(auto& profiler : m_profilers) {
			stream << '"' << profiler.first << "\",";
			profiler.second->save_total_and_snapshots(stream);
		}
	} catch(const std::exception& e) {
		logError("[Profiler::save_snapshots] Failed to save profiling results: ",
				 e.what());
	}
	return stream.str();
}

} // namespace mufflon