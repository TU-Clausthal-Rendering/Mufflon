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

struct Throughput {
	Spectrum weight;
	float guideWeight;
};

struct VertexSample : public math::PathSample {
	scene::Point origin;
	scene::materials::MediumHandle medium;	// TODO: fill this with life in the vertex itself
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
 * VERTEX_ALIGNMENT Number of bytes to which the vertices should be aligned (different
 *		for CPU and GPU).
 */
template < typename ExtensionT, int VERTEX_ALIGNMENT >
class PathVertex {
public:
	PathVertex() : m_type(Interaction::VOID) {}

	bool is_end_point() const {
		return m_type != Interaction::SURFACE;
	}
	bool is_orthographic() const {
		return m_type == Interaction::LIGHT_DIRECTIONAL
			|| m_type == Interaction::LIGHT_ENVMAP
			|| m_type == Interaction::CAMERA_ORTHO;
	}
	bool is_camera() const {
		return m_type == Interaction::CAMERA_PINHOLE
			|| m_type == Interaction::CAMERA_FOCUS
			|| m_type == Interaction::CAMERA_ORTHO;
	}

	// Get the position of the vertex. For orthographic vertices this
	// position is computed with respect to a referencePosition.
	scene::Point get_position(const scene::Point& referencePosition) const {
		if(m_type == Interaction::LIGHT_DIRECTIONAL
			|| m_type == Interaction::LIGHT_ENVMAP)
			return referencePosition - m_position * scene::MAX_SCENE_SIZE; // Go max entities out -- should be far enough away for shadow tests
		if(m_type == Interaction::CAMERA_ORTHO)
			return referencePosition; // TODO project to near plane
		return m_position;
	}

	// Get the position of the vertex. For orthographic vertices an assertion is issued
	scene::Point get_position() const {
		mAssertMsg(!is_orthographic(), "Implementation error. Orthogonal vertices have no position.");
		return m_position;
	}

	// The incident direction, or undefined for end vertices (may be abused by end vertices.
	scene::Direction get_incident_direction() const {
		mAssertMsg(!is_end_point(), "Incident direction for end points is not defined. Hope your code did not expect a meaningful value.");
		return m_incident;
	}

	// The per area PDF of the previous segment which generated this vertex
	AreaPdf get_incident_pdf() const { return m_incidentPdf; }

	// Get the 'cosθ' of the vertex for the purpose of AreaPdf::to_area_pdf(cosT, distSq);
	// This method ensures compatibility with any kind of interaction.
	// Non-hitable vertices simply return 0, surfaces return a real cosθ and
	// some things like the LIGHT_ENVMAP return special values for the compatibility.
	// connection: a direction with arbitrary orientation
	float get_geometrical_factor(const scene::Direction& connection) const {
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
				auto* alDesc = as<AreaLightDesc>(desc());
				return dot(connection, alDesc->normal);
			}
			case Interaction::SURFACE: {
				auto* surfDesc = as<SurfaceDesc>(desc());
				return dot(connection, surfDesc->tangentSpace.shadingN);
			}
		}
		return 0.0f;
	}

	// Get a normal if there is any. Otherwise returns a 0-vector.
	scene::Direction get_normal() const {
		if(m_type == Interaction::LIGHT_AREA) {
			auto* alDesc = as<AreaLightDesc>(desc());
			return alDesc->normal;
		}
		if(m_type == Interaction::SURFACE) {
			auto* surfDesc = as<SurfaceDesc>(desc());
			return surfDesc->tangentSpace.shadingN;
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
	math::EvalValue evaluate(const scene::Direction & excident,
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
				const DirLightDesc* desc = as<DirLightDesc>(this->desc());
				return lights::evaluate_dir(m_intensity, m_type==Interaction::LIGHT_ENVMAP,
									desc->dirPdf);
			}
			case Interaction::LIGHT_SPOT: {
				const SpotLightDesc* desc = as<SpotLightDesc>(this->desc());
				return lights::evaluate_spot(excident, m_intensity, desc->direction,
									desc->cosThetaMax, desc->cosFalloffStart);
			}
			case Interaction::LIGHT_AREA: {
				const AreaLightDesc* desc = as<AreaLightDesc>(this->desc());
				return lights::evaluate_area(excident, m_intensity, desc->normal);
			}
			case Interaction::CAMERA_PINHOLE: {
				const cameras::PinholeParams* desc = as<cameras::PinholeParams>(this->desc());
				cameras::ProjectionResult proj = pinholecam_project(*desc, excident);
				return math::EvalValue{ Spectrum{ proj.w }, 0.0f,
										proj.pdf, AngularPdf{ 0.0f } };
			}
			case Interaction::CAMERA_FOCUS: {
				const cameras::FocusParams* desc = as<cameras::FocusParams>(this->desc());
				cameras::ProjectionResult proj = focuscam_project(*desc, m_position, excident);
				return math::EvalValue{ Spectrum{proj.w}, 0.f,
										proj.pdf, AngularPdf{0.f} };
			}
			case Interaction::SURFACE: {
				const SurfaceDesc* desc = as<SurfaceDesc>(this->desc());
				return materials::evaluate(desc->tangentSpace, desc->params, m_incident,
										   excident, media, adjoint, merge);
			}
		}
		return math::EvalValue{};
	}

	// TODO: fractional pixel coords?
	Pixel get_pixel(const scene::Direction& excident) const {
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
	VertexSample sample(const scene::materials::Medium* media,
						const math::RndSet2_1& rndSet,
						bool adjoint = false
	) const {
		using namespace scene::lights;
		switch(m_type) {
			case Interaction::VOID: return VertexSample{};
			case Interaction::LIGHT_POINT: {
				auto lout = sample_light_dir_point(m_intensity, rndSet);
				return VertexSample{ { lout.flux, math::PathEventType::REFLECTED,
									   lout.dir.direction,
									   lout.dir.pdf, AngularPdf{0.0f} },
									 m_position };
			}
			case Interaction::LIGHT_DIRECTIONAL:
			case Interaction::LIGHT_ENVMAP: {
				// TODO: sample new positions on a boundary?
				const DirLightDesc* desc = as<DirLightDesc>(this->desc());
				return VertexSample{ { m_intensity, math::PathEventType::REFLECTED,
									   desc->direction, desc->dirPdf, AngularPdf{0.0f} },
									 m_position };
			}
			case Interaction::LIGHT_SPOT: {
				const SpotLightDesc* desc = as<SpotLightDesc>(this->desc());
				auto lout = sample_light_dir_spot(m_intensity, desc->direction, desc->cosThetaMax, desc->cosFalloffStart, rndSet);
				return VertexSample{ { lout.flux, math::PathEventType::REFLECTED,
									   lout.dir.direction, lout.dir.pdf, AngularPdf{0.0f} },
									 m_position };
			}
			case Interaction::LIGHT_AREA: {
				const AreaLightDesc* desc = as<AreaLightDesc>(this->desc());
				auto lout = sample_light_dir_area(m_intensity, desc->normal, rndSet);
				return VertexSample{ { lout.flux, math::PathEventType::REFLECTED,
									   lout.dir.direction, lout.dir.pdf, AngularPdf{0.0f} },
									 m_position };
			}
			case Interaction::CAMERA_PINHOLE: {
				const cameras::PinholeParams* desc = as<cameras::PinholeParams>(this->desc());
				cameras::Importon importon = pinholecam_sample_ray(*desc, m_position);
				return VertexSample{ { Spectrum{1.0f}, math::PathEventType::REFLECTED,
									   importon.dir.direction, importon.dir.pdf, AngularPdf{0.0f} },
									 m_position };
			}
			case Interaction::CAMERA_FOCUS: {
				const cameras::FocusParams* desc = as<cameras::FocusParams>(this->desc());
				cameras::Importon importon = focuscam_sample_ray(*desc, m_position, Pixel{ m_incident }, rndSet);
				return VertexSample{ { Spectrum{1.0f}, math::PathEventType::REFLECTED,
									   importon.dir.direction, importon.dir.pdf, AngularPdf{0.f} },
									 m_position };
			}
			case Interaction::SURFACE: {
				const SurfaceDesc* desc = as<SurfaceDesc>(this->desc());
				return VertexSample{ scene::materials::sample(
										desc->tangentSpace, desc->params,
										m_incident, media, rndSet, adjoint),
									 m_position };
			}
		}
		return VertexSample{};
	}

	// Compute the squared distance to the previous vertex. 0 if this is a start vertex.
	float get_incident_dist_sq(const void* pathMem) const {
		if(m_offsetToPath == 0xffff) return 0.0f;
		const PathVertex* prev = as<PathVertex>(as<u8>(pathMem) + m_offsetToPath);
		// The m_position is always a true position (the 'this' vertex is not an
		// end-point and can thus not be an orthogonal source).
		return lensq(prev->get_position(m_position) - m_position);
	}

	// Get the previous path vertex or nullptr if this is a start vertex.
	const PathVertex* previous(const void* pathMem) const {
		return m_offsetToPath == 0xffff ? nullptr : as<PathVertex>(as<u8>(pathMem) + m_offsetToPath);
	}

	// Access to the renderer dependent extension
	const ExtensionT& ext() const { return m_extension; }
	ExtensionT& ext()			  { return m_extension; }

	// Get the address of the interaction specific descriptor (aligned)
	// TODO: benchmark if internal alignment is worth it
	const void* desc() const { return as<u8>(this) + round_to_align(sizeof(PathVertex)); }

	/*
	* Compute the connection vector from path0 to path1 (non-normalized).
	* This is a non-trivial operation because of special cases like directional lights and
	* orthographic cammeras.
	*/
	static ei::Vec3 get_connection(const PathVertex& path0, const PathVertex& path1) {
		mAssert(is_connection_possible(path0, path1));
		if(path0.is_orthographic())
			return path0.m_position * scene::MAX_SCENE_SIZE;
		if(path1.is_orthographic())
			return -path1.m_position * scene::MAX_SCENE_SIZE;
		return path1.m_position - path0.m_position;
	}

	static bool is_connection_possible(const PathVertex& path0, const PathVertex& path1) {
		// Enumerate cases which are not connectible
		return !(
			path0.is_orthographic() && path1.is_orthographic()	// Two orthographic vertices
		 || path0.is_orthographic() && path1.is_camera()		// Orthographic light source with camera
		 || path0.is_camera() && path1.is_orthographic()
		);
		// TODO: camera clipping here? Seems to be the best location
	}

	Spectrum get_emission() const {
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
				const AreaLightDesc* desc = as<AreaLightDesc>(this->desc());
				return m_intensity; // TODO: / area?
			}
			case Interaction::SURFACE: {
				const SurfaceDesc* desc = as<SurfaceDesc>(this->desc());
				return scene::materials::emission(desc->params, m_incident);
			}
		}
		return Spectrum{0.0f};
	}

	Spectrum get_albedo() const {
		if(m_type == Interaction::SURFACE) {
			const SurfaceDesc* desc = as<SurfaceDesc>(this->desc());
			return scene::materials::albedo(desc->params);
		}
		// Area light source vertices are on surfaces with an albedo too.
		// However, it is likely that they are never asked for their albedo().
		mAssert(m_type != Interaction::LIGHT_AREA);
		return Spectrum{0.0f};
	}

	scene::accel_struct::RayIntersectionResult::HitID get_primitive_id() const {
		if(m_type == Interaction::SURFACE) {
			const SurfaceDesc* desc = as<SurfaceDesc>(this->desc());
			return desc->primitiveId;
		}
		// TODO id for area lights too
		return {-1, -1};
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
	static int create_void(void* mem, const void* previous,
		const ei::Ray& incidentRay
	) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_position = incidentRay.origin + incidentRay.direction * scene::MAX_SCENE_SIZE;
		vert->init_prev_offset(mem, previous);
		vert->m_type = Interaction::VOID;
		vert->m_incident = incidentRay.direction;
		vert->m_incidentPdf = AreaPdf { 0.0f };
		vert->m_extension = ExtensionT{};
		return round_to_align( sizeof(PathVertex) );
	}

	static int create_camera(void* mem, const void* previous,
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
			cameras::PinholeParams* desc = as<cameras::PinholeParams>(vert->desc());
			*desc = static_cast<const cameras::PinholeParams&>(camera);
			auto position = pinholecam_sample_position(*desc, pixel, rndSet);
			vert->m_position = position.position;
			vert->m_incidentPdf = position.pdf;
			return round_to_align( round_to_align(sizeof(PathVertex)) + sizeof(cameras::PinholeParams));
		}
		else if(camera.type == cameras::CameraModel::FOCUS) {
			vert->m_type = Interaction::CAMERA_FOCUS;
			cameras::FocusParams* desc = as<cameras::FocusParams>(vert->desc());
			*desc = static_cast<const cameras::FocusParams&>(camera);
			auto position = focuscam_sample_position(*desc, rndSet);
			vert->m_position = position.position;
			vert->m_incidentPdf = position.pdf;
			return round_to_align( round_to_align(sizeof(PathVertex)) + sizeof(cameras::FocusParams));
		}
		mAssertMsg(false, "Not implemented yet.");
		return 0;
	}

	static int create_light(void* mem, const void* previous,
		const scene::lights::Photon& lightSample,	// Positional sample for the starting point on a light source
		math::Rng& rng								// Only used for the incomplete vertices (env-map)
	) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_position = lightSample.pos.position;
		vert->init_prev_offset(mem, previous);
		vert->m_intensity = lightSample.intensity;
		vert->m_incidentPdf = lightSample.pos.pdf;
		vert->m_extension = ExtensionT{};
		switch(lightSample.type) {
			case scene::lights::LightType::POINT_LIGHT: {
				vert->m_type = Interaction::LIGHT_POINT;
				return round_to_align(sizeof(PathVertex));
			}
			case scene::lights::LightType::SPOT_LIGHT: {
				vert->m_type = Interaction::LIGHT_SPOT;
				SpotLightDesc* desc = as<SpotLightDesc>(vert->desc());
				desc->direction = lightSample.source_param.spot.direction;
				desc->cosThetaMax = lightSample.source_param.spot.cosThetaMax;
				desc->cosFalloffStart = lightSample.source_param.spot.cosFalloffStart;
				return round_to_align( round_to_align(sizeof(PathVertex)) + sizeof(SpotLightDesc));
			}
			case scene::lights::LightType::AREA_LIGHT_TRIANGLE:
			case scene::lights::LightType::AREA_LIGHT_QUAD:
			case scene::lights::LightType::AREA_LIGHT_SPHERE: {
				vert->m_type = Interaction::LIGHT_AREA;
				AreaLightDesc* desc = as<AreaLightDesc>(vert->desc());
				desc->normal = lightSample.source_param.area.normal;
				return round_to_align( round_to_align(sizeof(PathVertex)) + sizeof(AreaLightDesc));
			}
			case scene::lights::LightType::DIRECTIONAL_LIGHT: {
				vert->m_type = Interaction::LIGHT_DIRECTIONAL;
				DirLightDesc* desc = as<DirLightDesc>(vert->desc());
				desc->direction = lightSample.source_param.dir.direction;
				// Swap PDFs. The rules of sampling are reverted in directional
				// lights (first dir then pos is sampled). The vertex unifies
				// the view for MIS computation where the usage order is reverted.
				// Scale the pdfs to make sure later conversions lead to the original pdf.
				desc->dirPdf = lightSample.pos.pdf.to_angular_pdf(1.0f, ei::sq(scene::MAX_SCENE_SIZE));
				vert->m_incidentPdf = lightSample.source_param.dir.dirPdf.to_area_pdf(1.0f, ei::sq(scene::MAX_SCENE_SIZE));
				return round_to_align( round_to_align(sizeof(PathVertex)) + sizeof(DirLightDesc));
			}
			case scene::lights::LightType::ENVMAP_LIGHT: {
				vert->m_type = Interaction::LIGHT_ENVMAP;
				DirLightDesc* desc = as<DirLightDesc>(vert->desc());
				desc->direction = lightSample.source_param.dir.direction;
				desc->dirPdf = lightSample.pos.pdf.to_angular_pdf(1.0f, ei::sq(scene::MAX_SCENE_SIZE));
				vert->m_incidentPdf = lightSample.source_param.dir.dirPdf.to_area_pdf(1.0f, ei::sq(scene::MAX_SCENE_SIZE));
				return round_to_align( round_to_align(sizeof(PathVertex)) + sizeof(DirLightDesc));
			}
		}
	}

	static int create_surface(void* mem, const void* previous,
		const scene::accel_struct::RayIntersectionResult& hit,
		const scene::materials::MaterialDescriptorBase& material,
		const math::PositionSample& position,
		const scene::TangentSpace& tangentSpace,
		const scene::Direction& incident
	) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_position = position.position;
		vert->init_prev_offset(mem, previous);
		vert->m_type = Interaction::SURFACE;
		vert->m_incident = incident;
		vert->m_incidentPdf = position.pdf;
		vert->m_extension = ExtensionT{};
		SurfaceDesc* desc = as<SurfaceDesc>(vert->desc());
		desc->tangentSpace = tangentSpace;
		desc->primitiveId = hit.hitId;
		int size = scene::materials::fetch(material, hit.uv, &desc->params);
		return round_to_align( round_to_align(sizeof(PathVertex)) + sizeof(tangentSpace) + sizeof(scene::accel_struct::RayIntersectionResult::HitID) + size);
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
		scene::accel_struct::RayIntersectionResult::HitID primitiveId;
		scene::materials::ParameterPack params;
	};


	// The vertex position in world space. For orthographic end vertices
	// this is the main direction and not a position.
	scene::Point m_position;

	// Byte offset to the beginning of a path.
	u16 m_offsetToPath;

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

	// Area PDF of whatever created this vertex
	AreaPdf m_incidentPdf;

	// REMARK: currently 0 floats unused in 16-byte alignment
	ExtensionT m_extension;

	static constexpr int round_to_align(int s) {
		return (s + (VERTEX_ALIGNMENT-1)) & ~(VERTEX_ALIGNMENT-1);
	}

	// Private because the vertex is read-only by design (desc() is a helper for vertex creation only)
	void* desc() { return as<u8>(this) + round_to_align(sizeof(PathVertex)); }

	void init_prev_offset(void* mem, const void* previous) {
		std::size_t s = as<u8>(previous) - as<u8>(mem);
		mAssert(s <= 0xffff);
		m_offsetToPath = static_cast<u16>(s);
	}

	template<typename T, int A> 
	friend class PathVertexFactory;
};

}} // namespace mufflon::renderer