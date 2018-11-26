#pragma once

#include "core/scene/types.hpp"
#include "core/memory/residency.hpp"
#include "util/assert.hpp"
#include <string>

namespace mufflon {

// Forward declaration of device
enum class Device : unsigned char;

namespace cameras {

enum class CameraModel: i32 {
	PINHOLE,			// Infinite sharp pinhole camera
	FOCUS,				// Thin-lens physical related camera model
	ORTHOGRAPHIC,		// Orthographic projection

	NUM
};
#ifndef __CUDA_ARCH__
const std::string& to_string(CameraModel type);
#endif

// Basic header for camera parameter packs.
struct CameraParams {
	CameraModel type;
};

/*
 * Base class for all kinds of cameras. The basic camera includes an orthonormal
 * system for its view space and the position, but no ray logic.
 * Other more specific properties like the field of view are contained in the
 * according camera models.
 */
class Camera {
public:
	Camera() = default;
	Camera(ei::Vec3 position, ei::Vec3 dir, ei::Vec3 up,
		   float near = 1e-10f, float far = 1e10f) :
		m_position(std::move(position)),
		m_near(near),
		m_far(far)
	{
		dir = ei::normalize(dir);
		up = ei::normalize(up);
		if(ei::dot(dir, up) > 0.999f)
			throw std::runtime_error("View direction and up-vector are too close to each other");
		// Create orthonormal basis to determine view matrix
		const ei::Vec3 right = ei::normalize(ei::cross(up, dir));

		m_viewSpace = ei::Mat3x3{
			right.x, right.y, right.z,
			dir.x, dir.y, dir.z,
			up.x, up.y, up.z
		};
	}
	// Needs virtual destructor
	virtual ~Camera() = default;

	// The name of the camera as used by the scenario setup.
#ifndef __CUDACC__
	const std::string_view& get_name() const noexcept { return m_name; }
	void set_name(std::string_view name) { m_name = name; }
#endif

	const scene::Direction get_x_dir() const noexcept { return {m_viewSpace.m00, m_viewSpace.m01, m_viewSpace.m02}; }
	// The y-axis is up
	const scene::Direction get_up_dir() const noexcept { return {m_viewSpace.m10, m_viewSpace.m11, m_viewSpace.m12}; }
	// The z-axis is the view direction
	const scene::Direction get_view_dir() const noexcept { return {m_viewSpace.m20, m_viewSpace.m21, m_viewSpace.m22}; }
	const scene::Point& get_position() const noexcept { return m_position; }
	// The near clipping distance (plane)
	float get_near() const noexcept { return m_near; }
	void set_near(float n) noexcept { m_near = n; }
	// The far clipping distance (plane)
	float get_far() const noexcept { return m_far; }
	void set_far(float n) noexcept { m_far = n; }

	/*
	 * Translate the camera along its axis. To be used for interactive cameras
	 * leftRight: positive values to move right, negative values to move left (x-axis)
	 * upDown: positive values to move up, negative values to move down (y-axis)
	 * forBack: positive values to move forward, negative values to move backward (z-axis)
	 */
	void move(float leftRight, float upDown, float forBack) noexcept {
		m_position += leftRight * get_x_dir()
					+ upDown * get_up_dir()
					+ forBack * get_view_dir();
	}

	// Rotate around the x-axis.
	void rotate_up_down(Radians a) noexcept {
		m_viewSpace = ei::rotationX(a) * m_viewSpace;
	}
	// Rotate around the up direction (y-axis).
	void rotate_left_right(Radians a) noexcept {
		m_viewSpace = ei::rotationY(a) * m_viewSpace;
	}
	// Rotate around the view direction (z-axis).
	void roll(Radians a) noexcept {
		m_viewSpace = ei::rotationZ(a) * m_viewSpace;
	}

	/*
	 * Interface to obtain the architecture independent parameters required for sampling
	 * and evaluation. The outBuffer must have a size of at least get_parameter_pack_size().
	 * Each camera must implement a sample_ray and a project method (see cameras/sample.hpp
	 * for details).
	 */
	virtual void get_parameter_pack(CameraParams* outBuffer, Device dev) const = 0;

	// Get the required size of a parameter bundle.
	virtual std::size_t get_parameter_pack_size() const = 0;
protected:
	ei::Mat3x3 m_viewSpace;		// Orthonormal world->camera matrix (rows are axis, view direction is the third axis)
	scene::Point m_position;	// The central position for any projection
	float m_near {1e-10f};		// Optional near clipping distance
	float m_far {1e10f};		// Optional far clipping distance
private:
	std::string m_name;
};

/*
 * A RndSet is a fixed size set of random numbers which may be consumed by a camera
 * sampler. Note that material samplers have a different kind of RndSet.
 * The first pair (u0,u1) should be a high quality stratified sample.
 */
struct RndSet {
	float u0;	// In [0,1)
	float u1;	// In [0,1)
	float u2;	// In [0,1)
	float u3;	// In [0,1)

	RndSet(ei::Vec2 u01, ei::Vec2 u23) :
		u0(u01.x), u1(u01.y),
		u2(u23.x), u3(u23.y) {}
};

struct RaySample {
	// TODO: data layout? (currently set for GPU friendly padding)
	// TODO: per area pdf
	scene::Point origin {0.0f};			// Position on the near plane to start the ray
	AngularPdf pdf {0.0f};				// The camera sampling PDF
	scene::Direction excident {0.0f};	// The sampled direction
	float w {0.0f};						// The sensor response (equal to the PDF for some camera models)
};

struct ProjectionResult {
	Pixel coord {-1};					// The pixel in which the projection falls
	float pdf {0.0f};					// The camera sampling PDF
	float w {0.0f};						// The sensor response (equal to the PDF for some camera models)
};

} // namespace cameras

//template<> inline std::size_t predict_size<cameras::CameraParams>() {
//	mAssertMsg(false, "An instance of an unspecific camera should never be created!");
//	return 0;
//}

} // namespace mufflon