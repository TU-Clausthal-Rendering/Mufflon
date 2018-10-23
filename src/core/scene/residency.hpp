#pragma once

namespace mufflon::scene {

/// Contains the possible data locations of the scene.
enum class Residency {
	CPU,
	CUDA,
	OPENGL
};

} // namespace mufflon::scene