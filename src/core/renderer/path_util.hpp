﻿#pragma once

#include "util/types.hpp"
#include "core/memory/dyntype_memory.hpp"
#include "core/math/rng.hpp"
#include "core/math/sampling.hpp"
#include "core/scene/types.hpp"
#include "core/scene/materials/material_sampling.hpp"
#include "core/scene/lights/light_sampling.hpp"
#include "core/cameras/pinhole.hpp"
#include "core/cameras/focus.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include <ei/conversions.hpp>

namespace mufflon { namespace renderer {

enum class Interaction : u16 {
	VOID,				// The ray missed the scene (no intersection)
	SURFACE,			// A standard material interaction
	CAMERA_PINHOLE,		// A perspective camera start vertex
	CAMERA_FOCUS,		// A perspective camera with DoF blur effect
	CAMERA_ORTHO,		// An orthographic camera start vertex
	LIGHT_POINT,		// A point-light vertex
	LIGHT_DIRECTIONAL,	// A directional-light vertex
	LIGHT_ENVMAP,		// A directional-light from an environment map
	LIGHT_SPOT,			// A spot-light vertex
	LIGHT_AREA,			// An area-light vertex
};

// Braced-inherited initialization is only part of C++17...
struct VertexSample : public math::PathSample {
	VertexSample() = default;
	CUDA_FUNCTION VertexSample(math::PathSample pathSample, scene::Point origin,
				 scene::materials::MediumHandle medium) :
		math::PathSample(pathSample),
		origin(origin),
		medium(medium) {}

	scene::Point origin;
	scene::materials::MediumHandle medium;	// TODO: fill this with life in the vertex itself
};

// Return value helper
struct Connection {
	scene::Point v0;			// Origin / reference location of the first vertex
	float distance;				// Distance between v0 and v1
	scene::Direction dir;		// Vector v1 - v0, normalized
	float distanceSq;			// Squared distance
};

/*
 * The vertex is an abstraction layer between light/camera/material implementation
 * and a general rendering routine.
 * The type is very versatile as its size varyes for each different Interaction and Renderer.
 * A vertex is: a header (fixed size), renderer specific payload (template arguments) and
 * the descriptor for the interaction (size depends on interaction).
 *
 * ExtensionT: A single struct to extend the vertex for the use in a specialized renderer.
 *		@Alignment: the size of internal data is 40 byte. For GPU 16 byte alignment use
 *		the first 8 bytes of ExtensionT for scalar or Vec2 types. Only then use larger types
 *		like vec3/vec4.
 *		Automatic padding via alignas() will not run, because the compiler cannot see
 *		Vertex+ExtensionT as one type.
 */
template < typename ExtensionT >
class PathVertex {
public:
	CUDA_FUNCTION PathVertex() : m_type(Interaction::VOID) {}

	CUDA_FUNCTION bool is_end_point() const {
		return m_type != Interaction::SURFACE;
	}
	CUDA_FUNCTION bool is_orthographic() const {
		return m_type == Interaction::LIGHT_DIRECTIONAL
			|| m_type == Interaction::LIGHT_ENVMAP
			|| m_type == Interaction::CAMERA_ORTHO;
	}
	CUDA_FUNCTION bool is_camera() const {
		return m_type == Interaction::CAMERA_PINHOLE
			|| m_type == Interaction::CAMERA_FOCUS
			|| m_type == Interaction::CAMERA_ORTHO;
	}

	// Get the position of the vertex. For orthographic vertices this
	// position is computed with respect to a referencePosition.
	CUDA_FUNCTION scene::Point get_position(const scene::Point& referencePosition) const {
		if(m_type == Interaction::LIGHT_DIRECTIONAL
			|| m_type == Interaction::LIGHT_ENVMAP)
			return referencePosition - m_position * scene::MAX_SCENE_SIZE; // Go max entities out -- should be far enough away for shadow tests
		if(m_type == Interaction::CAMERA_ORTHO)
			return referencePosition; // TODO project to near plane
		return m_position;
	}

	// Get the position of the vertex. For orthographic vertices an assertion is issued
	CUDA_FUNCTION scene::Point get_position() const {
		mAssertMsg(!is_orthographic(), "Implementation error. Orthogonal vertices have no position.");
		return m_position;
	}

	// The incident direction, or undefined for end vertices (may be abused by end vertices.
	CUDA_FUNCTION scene::Direction get_incident_direction() const {
		mAssertMsg(!is_end_point(), "Incident direction for end points is not defined. Hope your code did not expect a meaningful value.");
		return m_incident;
	}

	// Get the 'cosθ' of the vertex for the purpose of AreaPdf::to_area_pdf(cosT, distSq);
	// This method ensures compatibility with any kind of interaction.
	// Non-hitable vertices simply return 0, surfaces return a real cosθ and
	// some things like the LIGHT_ENVMAP return special values for the compatibility.
	// connection: a direction with arbitrary orientation
	CUDA_FUNCTION float get_geometrical_factor(const scene::Direction& connection) const {
		switch(m_type) {
			// Non-hitable vertices
			case Interaction::VOID:
			case Interaction::CAMERA_PINHOLE:
			case Interaction::CAMERA_FOCUS:
			case Interaction::LIGHT_POINT:
			case Interaction::LIGHT_SPOT:
			case Interaction::LIGHT_DIRECTIONAL:
				return 0.0f;
			case Interaction::LIGHT_ENVMAP:
				// There is no geometrical conversion factor for hitting environment maps.
				// However, it is expected that the scene exit-distance is always the same
				// and can be used in both directions without changing the result of a
				// comparison.
				return 1.0f;
			case Interaction::LIGHT_AREA: {
				return dot(connection, m_desc.areaLight.normal);
			}
			case Interaction::SURFACE: {
				return dot(connection, m_desc.surface.tangentSpace.shadingN);
			}
		}
		return 0.0f;
	}

	// Get a normal if there is any. Otherwise returns a 0-vector.
	CUDA_FUNCTION scene::Direction get_normal() const {
		if(m_type == Interaction::LIGHT_AREA) {
			return m_desc.areaLight.normal;
		}
		if(m_type == Interaction::SURFACE) {
			return m_desc.surface.tangentSpace.shadingN;
		}
		return scene::Direction{0.0f};
	}

	// Get the sampling PDFs of this vertex (not defined, if
	// the vertex is an end point on a surface). Details at the members
	//AngularPdf get_forward_pdf() const { return m_pdfF; }
	//AngularPdf get_backward_pdf() const { return m_pdfB; }

	// Compute new values of the PDF and BxDF/outgoing radiance for a new
	// exident direction.
	// excident: direction pointing away from this vertex.
	// media: buffer with all media in the scene
	// adjoint: Is this a vertex on a light sub-path?
	// merge: Evaluation takes place in a merge?
	CUDA_FUNCTION math::EvalValue evaluate(const scene::Direction & excident,
							 const scene::materials::Medium* media,
							 bool adjoint = false, bool merge = false
	) const {
		using namespace scene;
		switch(m_type) {
			case Interaction::VOID: return math::EvalValue{};
			case Interaction::LIGHT_POINT: {
				return lights::evaluate_point(m_intensity);
			}
			case Interaction::LIGHT_DIRECTIONAL:
			case Interaction::LIGHT_ENVMAP: {
				return lights::evaluate_dir(m_intensity, m_type==Interaction::LIGHT_ENVMAP,
									m_desc.dirLight.dirPdf);
			}
			case Interaction::LIGHT_SPOT: {
				return lights::evaluate_spot(excident, m_intensity, m_desc.spotLight.direction,
									m_desc.spotLight.cosThetaMax, m_desc.spotLight.cosFalloffStart);
			}
			case Interaction::LIGHT_AREA: {
				return lights::evaluate_area(excident, m_intensity, m_desc.areaLight.normal);
			}
			case Interaction::CAMERA_PINHOLE: {
				cameras::ProjectionResult proj = pinholecam_project(m_desc.pinholeCam, excident);
				return math::EvalValue{ Spectrum{ proj.w }, 0.0f,
										proj.pdf, AngularPdf{ 0.0f } };
			}
			case Interaction::CAMERA_FOCUS: {
				cameras::ProjectionResult proj = focuscam_project(m_desc.focusCam, m_position, excident);
				return math::EvalValue{ Spectrum{proj.w}, 0.f,
										proj.pdf, AngularPdf{0.f} };
			}
			case Interaction::SURFACE: {
				return materials::evaluate(m_desc.surface.tangentSpace, m_desc.surface.mat(),
										   m_incident, excident, media, adjoint, merge);
			}
		}
		return math::EvalValue{};
	}

	// TODO: fractional pixel coords?
	CUDA_FUNCTION Pixel get_pixel(const scene::Direction& excident) const {
		if(m_type == Interaction::CAMERA_PINHOLE) {
			const cameras::PinholeParams* desc = as<cameras::PinholeParams>(this->desc());
			cameras::ProjectionResult proj = pinholecam_project(*desc, excident);
			return proj.coord;
		}
		if(m_type == Interaction::CAMERA_FOCUS) {
			const cameras::FocusParams* desc = as<cameras::FocusParams>(this->desc());
			cameras::ProjectionResult proj = focuscam_project(*desc, m_position, excident);
			return proj.coord;
		}
		// TODO: get from first vertex of path?
		return Pixel{-1};
	}

	/*
	 * Create a new outgoing direction. This method can be used in a loop
	 * to fully Monte Carlo integrate the rendering equation at this vertex.
	 */
	CUDA_FUNCTION VertexSample sample(const scene::materials::Medium* media,
						const math::RndSet2_1& rndSet,
						bool adjoint = false
	) const {
		using namespace scene::lights;
		switch(m_type) {
			case Interaction::VOID: return VertexSample{};
			case Interaction::LIGHT_POINT: {
				auto lout = sample_light_dir_point(m_intensity, rndSet);
				return VertexSample{ math::PathSample{ lout.flux, math::PathEventType::REFLECTED,
									   lout.dir.direction,
									   lout.dir.pdf, AngularPdf{0.0f} },
									 m_position, scene::materials::MediumHandle{} };
			}
			case Interaction::LIGHT_DIRECTIONAL:
			case Interaction::LIGHT_ENVMAP: {
				// TODO: sample new positions on a boundary?
				return VertexSample{ math::PathSample{ m_intensity, math::PathEventType::REFLECTED,
									   m_desc.dirLight.direction, m_desc.dirLight.dirPdf, AngularPdf{0.0f} },
									 m_position, scene::materials::MediumHandle{} };
			}
			case Interaction::LIGHT_SPOT: {
				auto lout = sample_light_dir_spot(m_intensity, m_desc.spotLight.direction, m_desc.spotLight.cosThetaMax, m_desc.spotLight.cosFalloffStart, rndSet);
				return VertexSample{ math::PathSample{ lout.flux, math::PathEventType::REFLECTED,
									   lout.dir.direction, lout.dir.pdf, AngularPdf{0.0f} },
									 m_position, scene::materials::MediumHandle{} };
			}
			case Interaction::LIGHT_AREA: {
				auto lout = sample_light_dir_area(m_intensity, m_desc.areaLight.normal, rndSet);
				return VertexSample{ math::PathSample{ lout.flux, math::PathEventType::REFLECTED,
									   lout.dir.direction, lout.dir.pdf, AngularPdf{0.0f} },
									 m_position, scene::materials::MediumHandle{} };
			}
			case Interaction::CAMERA_PINHOLE: {
				cameras::Importon importon = pinholecam_sample_ray(m_desc.pinholeCam, m_position);
				return VertexSample{ math::PathSample{ Spectrum{1.0f}, math::PathEventType::REFLECTED,
									   importon.dir.direction, importon.dir.pdf, AngularPdf{0.0f} },
									 m_position, scene::materials::MediumHandle{} };
			}
			case Interaction::CAMERA_FOCUS: {
				cameras::Importon importon = focuscam_sample_ray(m_desc.focusCam, m_position, Pixel{ m_incident }, rndSet);
				return VertexSample{ math::PathSample{ Spectrum{1.0f}, math::PathEventType::REFLECTED,
									   importon.dir.direction, importon.dir.pdf, AngularPdf{0.f} },
									 m_position, scene::materials::MediumHandle{} };
			}
			case Interaction::SURFACE: {
				return VertexSample{ scene::materials::sample(
										m_desc.surface.tangentSpace, m_desc.surface.mat(),
										m_incident, media, rndSet, adjoint),
									 m_position, scene::materials::MediumHandle{} };
			}
		}
		return VertexSample{};
	}

	// Compute the squared distance to the previous vertex. 0 if this is a start vertex.
	CUDA_FUNCTION float get_incident_dist_sq() const {
		if(m_offsetToPath == 0xffff) return 0.0f;
		const PathVertex* prev = as<PathVertex>(as<u8>(this) + m_offsetToPath);
		// The m_position is always a true position (the 'this' vertex is not an
		// end-point and can thus not be an orthogonal source).
		return lensq(prev->get_position(m_position) - m_position);
	}

	// Get the previous path vertex or nullptr if this is a start vertex.
	CUDA_FUNCTION const PathVertex* previous() const {
		return m_offsetToPath == 0xffff ? nullptr : as<PathVertex>(as<u8>(this) + m_offsetToPath);
	}

	// Access to the renderer dependent extension
	CUDA_FUNCTION ExtensionT& ext() const { return m_extension; }

	/*
	* Compute the connection vector from path0 to path1 (non-normalized).
	* This is a non-trivial operation because of special cases like directional lights and
	* orthographic cammeras.
	*/
	CUDA_FUNCTION static Connection get_connection(const PathVertex& path0, const PathVertex& path1) {
		mAssert(is_connection_possible(path0, path1));
		// Special cases
		if(path0.is_orthographic()) {	// p0 has no position
			mAssert(ei::approx(len(path0.m_position), 1.0f));
			return { path1.m_position - path0.m_position * scene::MAX_SCENE_SIZE, scene::MAX_SCENE_SIZE,
					 path0.m_position, ei::sq(scene::MAX_SCENE_SIZE) };
		}
		if(path1.is_orthographic()) {	// p1 has no position
			mAssert(ei::approx(len(path1.m_position), 1.0f));
			return { path0.m_position, scene::MAX_SCENE_SIZE,
					-path1.m_position, ei::sq(scene::MAX_SCENE_SIZE) };
		}
		// A normal connection (both vertices have a position)
		ei::Vec3 connection = path1.m_position - path0.m_position;
		float distSq = lensq(connection);
		float dist = sqrtf(distSq);
		return { path0.m_position, dist, sdiv(connection, dist), distSq };
	}

	CUDA_FUNCTION static bool is_connection_possible(const PathVertex& path0, const PathVertex& path1) {
		// Enumerate cases which are not connectible
		return !(
			path0.is_orthographic() && path1.is_orthographic()	// Two orthographic vertices
		 || path0.is_orthographic() && path1.is_camera()		// Orthographic light source with camera
		 || path0.is_camera() && path1.is_orthographic()
		);
		// TODO: camera clipping here? Seems to be the best location
	}

	CUDA_FUNCTION Spectrum get_emission() const {
		switch(m_type) {
			case Interaction::VOID:
			case Interaction::LIGHT_POINT:
			case Interaction::LIGHT_DIRECTIONAL:
			case Interaction::LIGHT_ENVMAP:
			case Interaction::LIGHT_SPOT:
			case Interaction::CAMERA_PINHOLE:
			case Interaction::CAMERA_FOCUS:
				return Spectrum{0.0f};
			case Interaction::LIGHT_AREA: {
				return m_intensity; // TODO: / area?
			}
			case Interaction::SURFACE: {
				return scene::materials::emission(m_desc.surface.mat(), m_desc.surface.tangentSpace.geoN, -m_incident);
			}
		}
		return Spectrum{0.0f};
	}

	CUDA_FUNCTION Spectrum get_albedo() const {
		if(m_type == Interaction::SURFACE) {
			return scene::materials::albedo(m_desc.surface.mat());
		}
		// Area light source vertices are on surfaces with an albedo too.
		// However, it is likely that they are never asked for their albedo().
		mAssert(m_type != Interaction::LIGHT_AREA);
		return Spectrum{0.0f};
	}

	CUDA_FUNCTION scene::PrimitiveHandle get_primitive_id() const {
		if(m_type == Interaction::SURFACE) {
			return m_desc.surface.primitiveId;
		}
		// TODO id for area lights too
		return {-1, -1};
	}

	// Get the surface parametrization (st) of the primitive. Only defined for
	// surface vertices
	CUDA_FUNCTION ei::Vec2 get_surface_params() const {
		if(m_type == Interaction::SURFACE) {
			return m_desc.surface.surfaceParams;
		}
		return {0.0f, 0.0f};
	}

	/* *************************************************************************
	 * Creation methods (factory)											   *
	 * Memory management of vertices is quite challenging, because its size	   *
	 * depends on the rendering algorithm and its interaction type.			   *
	 * We target to store vertices densly packet in local byte buffers. The	   *
	 * factory methods help with estimating sizes and creating the vertex	   *
	 * instances.															   *
	 * Return the size of the created vertex								   *
	************************************************************************* */
	CUDA_FUNCTION static int create_void(void* mem, const void* previous,
		const ei::Ray& incidentRay
	) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_position = incidentRay.origin + incidentRay.direction * scene::MAX_SCENE_SIZE;
		vert->init_prev_offset(mem, previous);
		vert->m_type = Interaction::VOID;
		vert->m_incident = incidentRay.direction;
		vert->m_extension = ExtensionT{};
		vert->ext().init(*vert, scene::Direction{0.0f}, 0.0f, 1.0f, AreaPdf { 0.0f });
		return round_to_align<VERTEX_ALIGNMENT>( sizeof(PathVertex) );
	}

	CUDA_FUNCTION static int create_camera(void* mem, const void* previous,
		const cameras::CameraParams& camera,
		const Pixel& pixel,
		const math::RndSet2& rndSet
	) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->init_prev_offset(mem, previous);
		vert->m_incident = ei::Vec3{
			static_cast<float>(pixel.x),
			static_cast<float>(pixel.y),
			0.f}; // will be (ab)used as pixel position storage
		vert->m_extension = ExtensionT{};
		if(camera.type == cameras::CameraModel::PINHOLE) {
			vert->m_type = Interaction::CAMERA_PINHOLE;
			vert->m_desc.pinholeCam = static_cast<const cameras::PinholeParams&>(camera);
			auto position = pinholecam_sample_position(vert->m_desc.pinholeCam, pixel, rndSet);
			vert->m_position = position.position;
			vert->ext().init(*vert, scene::Direction{0.0f}, 0.0f, 1.0f, position.pdf);
			return (int)round_to_align<8u>( this_size() + sizeof(cameras::PinholeParams));
		}
		else if(camera.type == cameras::CameraModel::FOCUS) {
			vert->m_type = Interaction::CAMERA_FOCUS;
			vert->m_desc.focusCam = static_cast<const cameras::FocusParams&>(camera);
			auto position = focuscam_sample_position(vert->m_desc.focusCam, rndSet);
			vert->m_position = position.position;
			vert->ext().init(*vert, scene::Direction{0.0f}, 0.0f, 1.0f, position.pdf);
			return (int)round_to_align<8u>( this_size() + sizeof(cameras::FocusParams));
		}
		mAssertMsg(false, "Not implemented yet.");
		return 0;
	}

	CUDA_FUNCTION static int create_light(void* mem, const void* previous,
		const scene::lights::Photon& lightSample,	// Positional sample for the starting point on a light source
		math::Rng& rng								// Only used for the incomplete vertices (env-map)
	) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_position = lightSample.pos.position;
		vert->init_prev_offset(mem, previous);
		vert->m_intensity = lightSample.intensity;
		vert->m_extension = ExtensionT{};
		switch(lightSample.type) {
			case scene::lights::LightType::POINT_LIGHT: {
				vert->m_type = Interaction::LIGHT_POINT;
				vert->ext().init(*vert, scene::Direction{0.0f}, 0.0f, 1.0f, lightSample.pos.pdf);
				return this_size();
			}
			case scene::lights::LightType::SPOT_LIGHT: {
				vert->m_type = Interaction::LIGHT_SPOT;
				vert->m_desc.spotLight.direction = lightSample.source_param.spot.direction;
				vert->m_desc.spotLight.cosThetaMax = lightSample.source_param.spot.cosThetaMax;
				vert->m_desc.spotLight.cosFalloffStart = lightSample.source_param.spot.cosFalloffStart;
				vert->ext().init(*vert, scene::Direction{0.0f}, 0.0f, 1.0f, lightSample.pos.pdf);
				return round_to_align<8u>( this_size() + sizeof(SpotLightDesc) );
			}
			case scene::lights::LightType::AREA_LIGHT_TRIANGLE:
			case scene::lights::LightType::AREA_LIGHT_QUAD:
			case scene::lights::LightType::AREA_LIGHT_SPHERE: {
				vert->m_type = Interaction::LIGHT_AREA;
				vert->m_desc.areaLight.normal = lightSample.source_param.area.normal;
				vert->ext().init(*vert, scene::Direction{0.0f}, 0.0f, 1.0f, lightSample.pos.pdf);
				return round_to_align<8u>( this_size() + sizeof(AreaLightDesc) );
			}
			case scene::lights::LightType::DIRECTIONAL_LIGHT: {
				vert->m_type = Interaction::LIGHT_DIRECTIONAL;
				vert->m_desc.dirLight.direction = lightSample.source_param.dir.direction;
				// Swap PDFs. The rules of sampling are reverted in directional
				// lights (first dir then pos is sampled). The vertex unifies
				// the view for MIS computation where the usage order is reverted.
				// Scale the pdfs to make sure later conversions lead to the original pdf.
				vert->m_desc.dirLight.dirPdf = lightSample.pos.pdf.to_angular_pdf(1.0f, ei::sq(scene::MAX_SCENE_SIZE));
				vert->ext().init(*vert, lightSample.source_param.dir.direction, scene::MAX_SCENE_SIZE, 1.0f,
					lightSample.source_param.dir.dirPdf.to_area_pdf(1.0f, ei::sq(scene::MAX_SCENE_SIZE)));
				return round_to_align<8u>( this_size() + sizeof(DirLightDesc) );
			}
			case scene::lights::LightType::ENVMAP_LIGHT: {
				vert->m_type = Interaction::LIGHT_ENVMAP;
				vert->m_desc.dirLight.direction = lightSample.source_param.dir.direction;
				vert->m_desc.dirLight.dirPdf = lightSample.pos.pdf.to_angular_pdf(1.0f, ei::sq(scene::MAX_SCENE_SIZE));
				vert->ext().init(*vert, lightSample.source_param.dir.direction, scene::MAX_SCENE_SIZE, 1.0f,
					lightSample.source_param.dir.dirPdf.to_area_pdf(1.0f, ei::sq(scene::MAX_SCENE_SIZE)));
				return round_to_align<8u>( this_size() + sizeof(DirLightDesc) );
			}
		}
	}

	CUDA_FUNCTION static int create_surface(void* mem, const void* previous,
		const scene::accel_struct::RayIntersectionResult& hit,
		const scene::materials::MaterialDescriptorBase& material,
		const scene::Point& position,
		const scene::TangentSpace& tangentSpace,
		const scene::Direction& incident,
		const float incidentDistance,
		const float incidentCosine,
		const AngularPdf prevPdf
	) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_position = position;
		vert->init_prev_offset(mem, previous);
		vert->m_type = Interaction::SURFACE;
		vert->m_incident = incident;
		vert->m_extension = ExtensionT{};
		vert->m_desc.surface.tangentSpace = tangentSpace;
		vert->m_desc.surface.primitiveId = hit.hitId;
		vert->m_desc.surface.surfaceParams = hit.surfaceParams.st;
		vert->ext().init(*vert, incident, incidentDistance, incidentCosine,
			prevPdf.to_area_pdf(incidentCosine, incidentDistance * incidentDistance));
		int size = scene::materials::fetch(material, hit.uv, &vert->m_desc.surface.mat());
		return (int)round_to_align<8u>( round_to_align<8u>(this_size() + sizeof(scene::TangentSpace)
			+ sizeof(scene::PrimitiveHandle) + sizeof(scene::accel_struct::SurfaceParametrization))
			+ size);
	}

private:
	struct AreaLightDesc {
		scene::Direction normal;
	};
	struct SpotLightDesc {
		ei::Vec3 direction;
		half cosThetaMax;
		half cosFalloffStart;
	};
	struct DirLightDesc {
		ei::Vec3 direction;
		AngularPdf dirPdf;
	};
	struct SurfaceDesc {
		scene::TangentSpace tangentSpace; // TODO: use packing?
		scene::PrimitiveHandle primitiveId;
		ei::Vec2 surfaceParams;
		//scene::materials::ParameterPack params;
		alignas(8) u8 matParams[scene::materials::MAX_MATERIAL_DESCRIPTOR_SIZE()];
		CUDA_FUNCTION const scene::materials::ParameterPack& mat() const { return *as<scene::materials::ParameterPack>(matParams); }
		CUDA_FUNCTION scene::materials::ParameterPack& mat() { return *as<scene::materials::ParameterPack>(matParams); }
	};


	// The vertex position in world space. For orthographic end vertices
	// this is the main direction and not a position.
	scene::Point m_position;

	// Byte offset to the beginning of a path.
	i16 m_offsetToPath;

	// Interaction type of this vertex (the descriptor at the end of the vertex depends on this).
	Interaction m_type;

	// Direction from which this vertex was reached.
	// May be zero-vector for start points or used otherwise.
	union {
		scene::Direction m_incident;		// For surface vertices
		Spectrum m_intensity;				// For light vertices
	};

	// PDF at this vertex in forward direction.
	// Non-zero for start points and zero for end points.
	//AngularPdf m_pdfF;

	// PDF in backward direction.
	// Zero for start points and non-zero for (real) end points. A path ending
	// on a surface will have a value of zero anyway.
	//AngularPdf m_pdfB;

	// REMARK: currently 1 floats unused in 16-byte alignment
	mutable ExtensionT m_extension;

	// Dynamic sized descriptor. Must be the last member!
	union Desc {
		CUDA_FUNCTION Desc() {}
		AreaLightDesc areaLight;
		SpotLightDesc spotLight;
		DirLightDesc dirLight;
		SurfaceDesc surface;
		cameras::PinholeParams pinholeCam;
		cameras::FocusParams focusCam;
	} m_desc;

	CUDA_FUNCTION void init_prev_offset(void* mem, const void* previous) {
		std::ptrdiff_t s = as<u8>(previous) - as<u8>(mem);
		mAssert(s <= 0xffff);
		m_offsetToPath = static_cast<i16>(s);
	}

	// Vertex size without the descriptor, the descriptor is 8byte aligned...
	static constexpr int this_size() {
		//return round_to_align<8u>(int(&m_desc - this));
		return (int)round_to_align<8u>( reinterpret_cast<std::ptrdiff_t>(&static_cast<PathVertex*>(nullptr)->m_desc) );
	}
};



// Interface example and dummy implementation for the vertex ExtensionT
struct VertexExtension {
	// The init function gets called at the end of vertex creation.
	CUDA_FUNCTION void init(const PathVertex<VertexExtension>& thisVertex,
							const scene::Direction& incident, const float incidentDistance,
							const float incidentCosine, const AreaPdf incidentPdf)
	{}

	// The update function gets called whenever the vertex was sampled.
	// thisVertex: Access to the vertex, for which 'this' is the payload
	//		hint: if you need information about the previous vertex use thisVertex.previous().
	// sample: The throughput, pdf and more of the current sampling event.
	CUDA_FUNCTION void update(const PathVertex<VertexExtension>& thisVertex,
							  const math::PathSample& sample) // TODO: ex cosine in path sample
	{}
};

}} // namespace mufflon::renderer