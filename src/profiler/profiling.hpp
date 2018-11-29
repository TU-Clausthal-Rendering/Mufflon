#pragma once

#include <memory>
#include <optional>
#include <ostream>
#include <string_view>
#include <type_traits>
#include <unordered_map>

namespace mufflon {

enum class ProfileLevel {
	LOW,
	HIGH,
	ALL
};

class ProfileState {
public:
	class ProfileScope {
	public:
		ProfileScope(ProfileState& state);
		ProfileScope(const ProfileScope&) = delete;
		ProfileScope(ProfileScope&&);
		ProfileScope& operator=(const ProfileScope&) = delete;
		ProfileScope& operator=(ProfileScope&&) = delete;
		~ProfileScope();
	private:
		ProfileState* m_state;
	};

	ProfileState(ProfileState*& active);
	virtual ~ProfileState() = default;

	virtual void create_snapshot() = 0;
	virtual void reset_sample() = 0;

	[[nodiscard]]
	ProfileScope start();
	void reset_all();
	void create_snapshot_all();
	std::ostream& save_snapshots(std::ostream& stream) const;
	std::ostream& save_current_state(std::ostream& stream) const;

	ProfileState* find_child(std::string_view name);
	ProfileState* add_child(std::string_view name, std::unique_ptr<ProfileState>&& child);

protected:
	virtual void start_sample() = 0;
	virtual void create_sample() = 0;
	virtual std::ostream& save_profiler_snapshots(std::ostream& stream) const = 0;
	virtual std::ostream& save_profiler_current_state(std::ostream& stream) const = 0;

private:
	// Actual method called by profilescope to initiate sample creation
	void add_sample();

	ProfileState*& m_activeRef;
	ProfileState* m_parent;
	std::unordered_map<std::string_view, std::unique_ptr<ProfileState>> m_children;
};

class Profiler {
public:
	template < class Prof >
	[[nodiscard]]
	std::optional<ProfileState::ProfileScope> start(std::string_view name, ProfileLevel level = ProfileLevel::HIGH) {
		if(m_enabled && static_cast<std::underlying_type_t<ProfileLevel>>(level)
		   <= static_cast<std::underlying_type_t<ProfileLevel>>(m_activation)) {
			if(m_active != nullptr) {
				// Use cascaded profiler
				ProfileState* state = m_active->find_child(name);
				if(state == nullptr)
					state = m_active->add_child(name, std::make_unique<Prof>(m_active));
				return state->start();
			} else {
				// Use top level profiler
				auto iter = m_profilers.find(name);
				if(iter == m_profilers.end())
					iter = m_profilers.emplace(name, std::make_unique<Prof>(m_active)).first;
				return iter->second->start();
			}
		}

		return std::nullopt;
	}

	static Profiler& instance();

	void reset_all();
	void reset(std::string_view name);
	void reset_from(std::string_view name);
	// Saves the current sample data and resets this and children's state
	void create_snapshot(std::string_view name);
	void create_snapshot_from(std::string_view name);
	void create_snapshot_all();
	void save_current_state(std::string_view path) const;
	void save_snapshots(std::string_view path) const;
	std::string save_current_state() const;
	std::string save_snapshots() const;

	// Sets the level of profiling
	constexpr ProfileLevel get_profile_level() const noexcept {
		return m_activation;
	}
	void set_profile_level(ProfileLevel level) noexcept {
		m_activation = level;
	}

	// Enables/disables profiling alltogether
	constexpr bool is_enabled() const noexcept {
		return m_enabled;
	}
	void set_enabled(bool enabled) noexcept {
		m_enabled = enabled;
	}

private:
	Profiler() = default;

	// Holds the top-level profilers
	bool m_enabled = false;
	ProfileState* m_active = nullptr;
	ProfileLevel m_activation = ProfileLevel::LOW;
	std::unordered_map<std::string_view, std::unique_ptr<ProfileState>> m_profilers;
};

} // namespace mufflon