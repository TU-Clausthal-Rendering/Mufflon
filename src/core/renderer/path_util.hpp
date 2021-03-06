﻿#pragma once

#include "util/types.hpp"
#include "core/memory/dyntype_memory.hpp"
#include "core/math/rng.hpp"
#include "core/math/sampling.hpp"
#include "core/scene/types.hpp"
#include "core/scene/materials/material_sampling.hpp"
#include "core/scene/lights/light_sampling.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include "core/cameras/pinhole.hpp"
#include "core/cameras/focus.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include <ei/conversions.hpp>

namespace mufflon { namespace renderer {

enum class Interaction : u16 {
	VOID,				// The ray missed the scene (no intersection)
	VIRTUAL,			// Virtual vertices are used for random hit evaluations:
						//	A random hit is a connection to a virtual vertex whose backward
						//	pdf is the pdf to create the NEE vertex.
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

inline CUDA_FUNCTION bool is_end_point(Interaction type) {
	return type != Interaction::SURFACE;
}
inline CUDA_FUNCTION bool is_surface(Interaction type) {
	return type == Interaction::SURFACE
		|| type == Interaction::LIGHT_AREA;
}
inline CUDA_FUNCTION bool is_orthographic(Interaction type) {
	return type == Interaction::LIGHT_DIRECTIONAL
		|| type == Interaction::LIGHT_ENVMAP
		|| type == Interaction::CAMERA_ORTHO
		|| type == Interaction::VOID;
}
inline CUDA_FUNCTION bool is_camera(Interaction type) {
	return type == Interaction::CAMERA_PINHOLE
		|| type == Interaction::CAMERA_FOCUS
		|| type == Interaction::CAMERA_ORTHO;
}

inline CUDA_FUNCTION bool is_hitable(Interaction type) {
	return type == Interaction::SURFACE
		|| type == Interaction::LIGHT_AREA
		|| type == Interaction::LIGHT_ENVMAP
		|| type == Interaction::VOID;
}

// Braced-inherited initialization is only part of C++17...
struct VertexSample : public math::PathSample {
	VertexSample() = default;
	inline CUDA_FUNCTION VertexSample(math::PathSample pathSample, scene::Point origin,
				 scene::materials::MediumHandle medium) :
		math::PathSample(pathSample),
		origin(origin),
		medium(medium) {}

	scene::Point origin;
	scene::materials::MediumHandle medium;	// TODO: fill this with life in the vertex itself
};

// Return value helper
struct ConnectionDir {
	scene::Direction dir;		// Vector v1 - v0, normalized
	float distanceSq;			// Squared distance or 1 for connections to orthographic vertices
};
struct Connection : public ConnectionDir {
	scene::Point v0;			// Origin / reference location of the first vertex
	float distance;				// Distance between v0 and v1 or MAX_SCENE_SIZE for orthographic connections

	inline CUDA_FUNCTION Connection(const scene::Direction& dir, float distanceSq,
							 const scene::Point& v0, float distance) :
		ConnectionDir{dir, distanceSq},
		v0{v0},
		distance{distance}
	{}
};

struct EmissionValue : public math::SampleValue, public scene::lights::LightPdfs {
	inline CUDA_FUNCTION EmissionValue(const Spectrum& radiance, AngularPdf pdf, AreaPdf emitPdf, AreaPdf connectPdf) :
		math::SampleValue{radiance, pdf},
		scene::lights::LightPdfs{emitPdf, connectPdf}
	{}

	inline CUDA_FUNCTION EmissionValue() :
		math::SampleValue{Spectrum{0.0f}, AngularPdf{0.0f}},
		scene::lights::LightPdfs{AreaPdf{0.0f}, AreaPdf{0.0f}}
	{}
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
	inline CUDA_FUNCTION PathVertex() : m_type(Interaction::VOID) {}

	inline CUDA_FUNCTION bool is_end_point() const { return renderer::is_end_point(m_type); }
	inline CUDA_FUNCTION bool is_surface() const { return renderer::is_surface(m_type); }
	inline CUDA_FUNCTION bool is_orthographic() const { return renderer::is_orthographic(m_type); }
	inline CUDA_FUNCTION bool is_camera() const { return renderer::is_camera(m_type); }
	inline CUDA_FUNCTION bool is_hitable() const { return renderer::is_hitable(m_type); }
	inline CUDA_FUNCTION Interaction get_type() const { return m_type; }

	// Get the position of the vertex. For orthographic vertices this
	// position is computed with respect to a referencePosition.
	inline CUDA_FUNCTION scene::Point get_position(const scene::Point& referencePosition) const {
		if(m_type == Interaction::LIGHT_DIRECTIONAL
			|| m_type == Interaction::LIGHT_ENVMAP)
			return referencePosition - get_incident_direction() * scene::MAX_SCENE_SIZE; // Go max entities out -- should be far enough away for shadow tests
		if(m_type == Interaction::CAMERA_ORTHO)
			return referencePosition; // TODO project to near plane
		return m_position;
	}

	// Get the position of the vertex. For orthographic vertices zero is returned
	inline CUDA_FUNCTION scene::Point get_position() const {
		if(is_orthographic()) return scene::Point{0.0f};
		return m_position;
	}

	// The incident direction, or undefined for end vertices (may be abused by end vertices).
	// The direction points towards the surface.
	inline CUDA_FUNCTION scene::Direction get_incident_direction() const {
		float z = ei::sgn(m_incidentEncoded.z) * sqrt(ei::clamp(1.0f - m_incidentEncoded.x * m_incidentEncoded.x - m_incidentEncoded.y * m_incidentEncoded.y, 0.0f, 1.0f));
		return {m_incidentEncoded.x, m_incidentEncoded.y, z};
	}
	inline CUDA_FUNCTION void set_incident_direction(const scene::Direction& incident, const float distance) {
		m_incidentEncoded.x = incident.x;
		m_incidentEncoded.y = incident.y;
		m_incidentEncoded.z = distance * ei::sgn(incident.z);
	}

	// Get the 'cosθ' of the vertex for the purpose of AreaPdf::to_area_pdf(cosθ, distSq);
	// This method ensures compatibility with any kind of interaction.
	// connection: a direction with arbitrary orientation
	inline CUDA_FUNCTION float get_geometric_factor(const scene::Direction& connection) const {
		switch(m_type) {
			case Interaction::LIGHT_AREA: {
				return dot(connection, m_incidentEncoded);
			}
			case Interaction::SURFACE:
			case Interaction::VIRTUAL: {
				return dot(connection, m_desc.surface.tangentSpace.shadingN);
			}
			case Interaction::LIGHT_ENVMAP:
			case Interaction::VOID: {
				return 1.0f;
			}
			default: mAssert(false); break;
		}
		return 0.0f;
	}

	// Get a normal if there is any. Otherwise returns a 0-vector.
	inline CUDA_FUNCTION scene::Direction get_normal() const {
		if(m_type == Interaction::LIGHT_AREA) {
			return m_incidentEncoded;
		}
		if(m_type == Interaction::SURFACE) {
			return m_desc.surface.tangentSpace.shadingN;
		}
		return scene::Direction{0.0f};
	}
	inline CUDA_FUNCTION scene::Direction get_geometric_normal() const {
		if(m_type == Interaction::LIGHT_AREA) {
			return m_incidentEncoded;
		}
		if(m_type == Interaction::SURFACE) {
			return m_desc.surface.tangentSpace.geoN;
		}
		return scene::Direction{0.0f};
	}

	inline CUDA_FUNCTION const scene::TangentSpace* get_tangent_space() const {
		if(m_type == Interaction::SURFACE) {
			return &m_desc.surface.tangentSpace;
		}
		return nullptr;
	}

	// Convert a sampling pdf (areaPdf for orthographic vertices, angular otherwise)
	// into an areaPdf at this vertex.
	struct PdfConversionResult{ AreaPdf pdf; float geoFactor; };
	inline CUDA_FUNCTION PdfConversionResult
	convert_pdf(Interaction sourceType, AngularPdf samplePdf,
				const ConnectionDir& connection) const {
		if(sourceType == Interaction::VIRTUAL)
			return { AreaPdf{ float(samplePdf) }, 1.0f };
		// For orthographic vertices the pdf does not change with a distance.
		// It is simply projected directly to the surface.
		bool orthoSource = renderer::is_orthographic(sourceType);
		bool orthoTarget = this->is_orthographic();
		float geoFactor = this->get_geometric_factor(connection.dir);
		if(orthoSource || orthoTarget) {
			return { AreaPdf{ float(samplePdf) * ei::abs(geoFactor) }, geoFactor };
		}
		return { samplePdf.to_area_pdf(geoFactor, connection.distanceSq), geoFactor };
	}

	// Get the path length in segments up to this vertex.
	inline CUDA_FUNCTION int get_path_len() const { return m_pathLen; }
	// Overwrite path length (e.g. if the previous element is not set/changed for this vertex)
	inline CUDA_FUNCTION void set_path_len(int l) { m_pathLen = l; }

	// Get the sampling PDFs of this vertex (not defined, if
	// the vertex is an end point on a surface). Details at the members
	//AngularPdf get_forward_pdf() const { return m_pdfF; }
	//AngularPdf get_backward_pdf() const { return m_pdfB; }

	// Compute new values of the PDF and BxDF/outgoing radiance for a new
	// exident direction.
	// excident: direction pointing away from this vertex.
	// media: buffer with all media in the scene
	// coord: [out] Pixel coordinate if a projection was done.
	//		Otherwise the input value will be preserved.
	// adjoint: Is this a vertex on a light sub-path?
	// lightNormal: !=0 -> Evaluation takes place in a merge.
	//		For shading normal correction in merges it is necessary to know the
	//		cosines at the other vertex. Note that merges are always evaluated
	//		at the view path vertex.
	inline CUDA_FUNCTION math::EvalValue evaluate(const scene::Direction& excident,
							 const scene::materials::Medium* media,
							 Pixel& coord,
							 bool adjoint = false,
							 const scene::Direction* lightNormal = nullptr
	) const {
		using namespace scene;
		switch(m_type) {
			case Interaction::VOID: return math::EvalValue{};
			case Interaction::LIGHT_POINT: {
				return lights::evaluate_point(m_incidentEncoded);
			}
			case Interaction::LIGHT_DIRECTIONAL:
			case Interaction::LIGHT_ENVMAP: {
				return lights::evaluate_dir(m_desc.dirLight.irradiance,
											m_type==Interaction::LIGHT_ENVMAP,
											m_desc.dirLight.projSceneArea);
			}
			case Interaction::LIGHT_SPOT: {
				return lights::evaluate_spot(excident, m_desc.spotLight.intensity, m_incidentEncoded,
									m_desc.spotLight.cosThetaMax, m_desc.spotLight.cosFalloffStart);
			}
			case Interaction::LIGHT_AREA: {
				return lights::evaluate_area(excident, m_desc.areaLight.intensity, m_incidentEncoded);
			}
			case Interaction::CAMERA_PINHOLE: {
				// TODO: fractional pixel coords?
				cameras::ProjectionResult proj = pinholecam_project(m_desc.pinholeCam, excident);
				coord = proj.coord;
				return math::EvalValue{ Spectrum{ proj.w }, 1.0f,
										{ proj.pdf, AngularPdf{ 0.0f } } };
			}
			case Interaction::CAMERA_FOCUS: {
				// TODO: fractional pixel coords?
				cameras::ProjectionResult proj = focuscam_project(m_desc.focusCam, m_position, excident);
				coord = proj.coord;
				return math::EvalValue{ Spectrum{ proj.w }, 1.0f,
										{ proj.pdf, AngularPdf{ 0.0f } } };
			}
			case Interaction::SURFACE: {
				return materials::evaluate(m_desc.surface.tangentSpace, m_desc.surface.material,
										   get_incident_direction(), excident, media, adjoint, lightNormal);
			}
			default: mAssert(false); break;
		}
		return math::EvalValue{};
	}

	/*
	 * Create a new outgoing direction. This method can be used in a loop
	 * to fully Monte Carlo integrate the rendering equation at this vertex.
	 */
	inline CUDA_FUNCTION VertexSample sample(const ei::Box& sceneBounds,
						const scene::materials::Medium* media,
						const math::RndSet2_1& rndSet,
						bool adjoint = false
	) const {
		using namespace scene::lights;
		switch(m_type) {
			case Interaction::VOID: return VertexSample{};
			case Interaction::LIGHT_POINT: {
				auto lout = sample_light_dir_point(m_incidentEncoded, rndSet);
				return VertexSample{ math::PathSample{ lout.flux, math::PathEventType::SOURCE,
									   lout.dir.direction,
									   {lout.dir.pdf, AngularPdf{0.0f}} },
									 m_position, m_desc.pointLight.medium };
			}
			case Interaction::LIGHT_DIRECTIONAL:
			case Interaction::LIGHT_ENVMAP: {
				// Sample new positions on a boundary
				math::PositionSample pos = math::sample_position(m_incidentEncoded, sceneBounds, rndSet.u0, rndSet.u1);
				Spectrum flux = m_desc.dirLight.irradiance / float(pos.pdf);
				return VertexSample{ math::PathSample{ flux, math::PathEventType::ORTHO_SOURCE,
									   m_incidentEncoded, {AngularPdf{float(pos.pdf)}, AngularPdf{0.0f}} },
									 pos.position, m_desc.dirLight.medium };
			}
			case Interaction::LIGHT_SPOT: {
				auto lout = sample_light_dir_spot(m_desc.spotLight.intensity, m_incidentEncoded, m_desc.spotLight.cosThetaMax, m_desc.spotLight.cosFalloffStart, rndSet);
				return VertexSample{ math::PathSample{ lout.flux, math::PathEventType::SOURCE,
									   lout.dir.direction, {lout.dir.pdf, AngularPdf{0.0f}} },
									 m_position, m_desc.spotLight.medium };
			}
			case Interaction::LIGHT_AREA: {
				auto lout = sample_light_dir_area(m_desc.areaLight.intensity, m_incidentEncoded, rndSet);
				return VertexSample{ math::PathSample{ lout.flux, math::PathEventType::SOURCE,
									   lout.dir.direction, {lout.dir.pdf, AngularPdf{0.0f}} },
									 m_position, m_desc.areaLight.medium };
			}
			case Interaction::CAMERA_PINHOLE: {
				cameras::Importon importon = pinholecam_sample_ray(m_desc.pinholeCam, Pixel{ m_incidentEncoded }, rndSet);
				return VertexSample{ math::PathSample{ Spectrum{1.0f}, math::PathEventType::SOURCE,
									   importon.dir.direction, {importon.dir.pdf, AngularPdf{0.0f}} },
									 m_position, m_desc.pinholeCam.mediumIndex };
			}
			case Interaction::CAMERA_FOCUS: {
				cameras::Importon importon = focuscam_sample_ray(m_desc.focusCam, m_position, Pixel{ m_incidentEncoded }, rndSet);
				return VertexSample{ math::PathSample{ Spectrum{1.0f}, math::PathEventType::SOURCE,
									   importon.dir.direction, {importon.dir.pdf, AngularPdf{0.0f}} },
									 m_position, m_desc.focusCam.mediumIndex };
			}
			case Interaction::SURFACE: {
				math::PathSample s = scene::materials::sample(
										m_desc.surface.tangentSpace, m_desc.surface.material,
										get_incident_direction(), media, rndSet, adjoint);
				float side = dot(s.excident, m_desc.surface.tangentSpace.geoN);
				return VertexSample{ s, m_position, m_desc.surface.material.get_medium(side) };
			}
			default: mAssert(false); break;
		}
		return VertexSample{};
	}

	// Compute the squared distance to the previous vertex. 0 if this is a start vertex.
	inline CUDA_FUNCTION float get_incident_dist_sq() const {
		if(is_end_point()) return 0.0f;
		return ei::sq(m_incidentEncoded.z);
	}
	inline CUDA_FUNCTION float get_incident_dist() const {
		if(is_end_point()) return 0.0f;
		return ei::abs(m_incidentEncoded.z);
	}

	inline CUDA_FUNCTION ConnectionDir get_incident_connection() const {
		return { get_incident_direction(), get_incident_dist_sq() };
	}

	// Get the previous path vertex or nullptr if this is a start vertex.
	inline CUDA_FUNCTION const PathVertex* previous() const {
		return m_previous;
	}

	// Access to the renderer dependent extension
	inline CUDA_FUNCTION ExtensionT& ext() const { return m_extension; }

	/*
	* Compute the connection vector from path0 to path1.
	* This is a non-trivial operation because of special cases like directional lights and
	* orthographic cammeras.
	*/
	inline CUDA_FUNCTION static Connection get_connection(const PathVertex& path0, const PathVertex& path1) {
		mAssert(is_connection_possible(path0, path1));
		// Special cases
		if(path0.is_orthographic()) {	// p0 has no position
			return { path0.m_incidentEncoded, 1.0f,
					 path1.m_position - path0.m_incidentEncoded * scene::MAX_SCENE_SIZE, scene::MAX_SCENE_SIZE };
		}
		if(path1.is_orthographic()) {	// p1 has no position
			return { -path1.m_incidentEncoded, 1.0f,
					  path0.m_position, scene::MAX_SCENE_SIZE };
		}
		// A normal connection (both vertices have a position)
		ei::Vec3 connection = path1.m_position - path0.m_position;
		float distSq = lensq(connection);
		float dist = sqrtf(distSq);
		return { sdiv(connection, dist), distSq, path0.m_position, dist };
	}

	inline CUDA_FUNCTION static bool is_connection_possible(const PathVertex& path0, const PathVertex& path1) {
		// Enumerate cases which are not connectible
		return !(
			(path0.is_orthographic() && path1.is_orthographic())	// Two orthographic vertices
		 || (path0.is_orthographic() && path1.is_camera())			// Orthographic light source with camera
		 || (path0.is_camera() && path1.is_orthographic())
		);
		// TODO: camera clipping here? Seems to be the best location
	}

	// connectPdf: compute the emit_pdf() for area lights, otherwise compute connect_pdf()
	inline CUDA_FUNCTION EmissionValue get_emission(const scene::SceneDescriptor<CURRENT_DEV>& scene, const ei::Vec3& prevPosition) const {
		switch(m_type) {
			case Interaction::VOID: {
				math::EvalValue background = evaluate_background(scene.lightTree.background, m_incidentEncoded);
				AngularPdf backtracePdf { 0.0f };
				AreaPdf startPdf { 0.0f };
				if(background.value != 0.0f) {
					backtracePdf = AngularPdf{ 1.0f / math::projected_area(m_incidentEncoded, scene.aabb) };
					startPdf = background_pdf(scene.lightTree, background);
				}
				return { background.value, backtracePdf, startPdf, startPdf };
			}
			case Interaction::LIGHT_POINT:
			case Interaction::LIGHT_DIRECTIONAL:
			case Interaction::LIGHT_ENVMAP:
			case Interaction::LIGHT_SPOT:
			case Interaction::CAMERA_PINHOLE:
			case Interaction::CAMERA_FOCUS:
				return EmissionValue{};
			case Interaction::LIGHT_AREA: {
				// If an area light is hit, a surface vertex should be created.
				// Area light vertices do not store the incident direction (they are
				// start points).
				mAssert(false);
				return EmissionValue{};
			}
			case Interaction::SURFACE: {
				math::SampleValue matValue = scene::materials::emission(m_desc.surface.material, m_desc.surface.tangentSpace.geoN, -get_incident_direction());
				scene::lights::LightPdfs startPdf { AreaPdf{0.0f}, AreaPdf{0.0f} };
				if(matValue.value != 0.0f) {
					startPdf = scene::lights::light_pdf(scene.lightTree, m_desc.surface.primitiveId,
										 m_desc.surface.surfaceParams, prevPosition);
				}
				return { matValue.value, matValue.pdf, startPdf.emitPdf, startPdf.connectPdf };
			}
			default: mAssert(false); break;
		}
		return EmissionValue{};
	}

	inline CUDA_FUNCTION Spectrum get_albedo() const {
		if(m_type == Interaction::SURFACE) {
			return scene::materials::albedo(m_desc.surface.material);
		}
		// Area light source vertices are on surfaces with an albedo too.
		// However, it is likely that they are never asked for their albedo().
		mAssert(m_type != Interaction::LIGHT_AREA);
		return Spectrum{0.0f};
	}

	inline CUDA_FUNCTION scene::materials::MediumHandle get_medium(const scene::Direction& dir) const {
		switch(m_type) {
			case Interaction::VOID: return scene::materials::MediumHandle{};
			case Interaction::LIGHT_POINT: return m_desc.pointLight.medium;
			case Interaction::LIGHT_DIRECTIONAL:
			case Interaction::LIGHT_ENVMAP: return m_desc.dirLight.medium;
			case Interaction::LIGHT_SPOT: return m_desc.spotLight.medium;
			case Interaction::LIGHT_AREA: return m_desc.areaLight.medium;
			case Interaction::CAMERA_PINHOLE: return m_desc.pinholeCam.mediumIndex;
			case Interaction::CAMERA_FOCUS: return m_desc.focusCam.mediumIndex;
			case Interaction::SURFACE: {
				float side = dot(dir, m_desc.surface.tangentSpace.geoN);
				return m_desc.surface.material.get_medium(side);
			}
			default: mAssert(false); break;
		}
		return scene::materials::MediumHandle{};
	}

	inline CUDA_FUNCTION scene::PrimitiveHandle get_primitive_id() const {
		if(m_type == Interaction::SURFACE) {
			return m_desc.surface.primitiveId;
		}
		// TODO id for area lights too
		return {-1, -1};
	}

	// Get the surface parametrization (st) of the primitive. Only defined for
	// surface vertices
	inline CUDA_FUNCTION ei::Vec2 get_surface_params() const {
		if(m_type == Interaction::SURFACE) {
			return m_desc.surface.surfaceParams;
		}
		return {0.0f, 0.0f};
	}

	// TODO: for other vertices than surfaces
	inline CUDA_FUNCTION float get_pdf_max() const {
		if(m_type == Interaction::SURFACE) {
			return pdf_max(m_desc.surface.material);
		}
		return 0.0f;
	}

	// Get the ratio of incident refraction index to trancemitted IOR if applicable
	struct RefractionInfo {
		float eta;		// Ratio of incident IOR to trancemitted
		float inCos;	// Incident cosine
	//	float outCos;	// Outgoing cosine after Fresnel
	};
	inline CUDA_FUNCTION RefractionInfo get_eta(const scene::materials::Medium* media) const {
		if(m_type == Interaction::SURFACE) {
			float inCos = -dot(get_incident_direction(), m_desc.surface.tangentSpace.shadingN);
			scene::materials::MediumHandle mIn = m_desc.surface.material.get_medium(inCos);
			scene::materials::MediumHandle mOut = m_desc.surface.material.get_medium(-inCos);
			float eta = media[mIn].get_refraction_index().x / media[mOut].get_refraction_index().x;
			return { eta, inCos };
		//	float outCos = sqrt(ei::max(0.0f, 1.0f - eta * eta * (1.0f - inCos * inCos)));
		//	return { eta, inCos, outCos };
		}
		return { 1.0f, 1.0f };
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
	inline CUDA_FUNCTION static int create_camera(void* mem, const void* previous,
		const cameras::CameraParams& camera,
		const Pixel& pixel,
		const math::RndSet2& rndSet
	) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_previous = mem != previous ? static_cast<const PathVertex*>(previous) : nullptr;
		vert->m_pathLen = previous ? static_cast<const PathVertex*>(previous)->m_pathLen + 1 : 0;
		vert->m_incidentEncoded = ei::Vec3{
			static_cast<float>(pixel.x),
			static_cast<float>(pixel.y),
			0.f}; // will be (ab)used as pixel position storage
		vert->m_extension = ExtensionT{};
		if(camera.type == cameras::CameraModel::PINHOLE) {
			vert->m_type = Interaction::CAMERA_PINHOLE;
			vert->m_desc.pinholeCam = static_cast<const cameras::PinholeParams&>(camera);
			auto position = pinholecam_sample_position(vert->m_desc.pinholeCam);
			vert->m_position = position.position;
			vert->ext().init(*vert, AreaPdf::infinite(), AngularPdf::infinite(), 1.0f);
			return (int)round_to_align<8u>( this_size() + sizeof(cameras::PinholeParams));
		}
		else if(camera.type == cameras::CameraModel::FOCUS) {
			vert->m_type = Interaction::CAMERA_FOCUS;
			vert->m_desc.focusCam = static_cast<const cameras::FocusParams&>(camera);
			auto position = focuscam_sample_position(vert->m_desc.focusCam, rndSet);
			vert->m_position = position.position;
			vert->ext().init(*vert, position.pdf, AngularPdf::infinite(), 1.0f);
			return (int)round_to_align<8u>( this_size() + sizeof(cameras::FocusParams));
		}
		mAssertMsg(false, "Not implemented yet.");
		return 0;
	}

	inline CUDA_FUNCTION static int create_light(void* mem, const void* previous,
		const scene::lights::Emitter& lightSample	// Positional sample for the starting point on a light source
	) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_position = lightSample.initVec;
		vert->m_previous = mem != previous ? static_cast<const PathVertex*>(previous) : nullptr;
		vert->m_pathLen = previous ? static_cast<const PathVertex*>(previous)->m_pathLen + 1 : 0;
		vert->m_extension = ExtensionT{};
		switch(lightSample.type) {
			case scene::lights::LightType::POINT_LIGHT: {
				vert->m_type = Interaction::LIGHT_POINT;
				vert->m_incidentEncoded = lightSample.intensity;
				vert->m_desc.pointLight.medium = lightSample.mediumIndex;
				vert->ext().init(*vert, AreaPdf::infinite(), AngularPdf::infinite(), lightSample.pChoice);
				return this_size();
			}
			case scene::lights::LightType::SPOT_LIGHT: {
				vert->m_type = Interaction::LIGHT_SPOT;
				vert->m_incidentEncoded = lightSample.source_param.spot.direction;
				vert->m_desc.spotLight.intensity = lightSample.intensity;
				vert->m_desc.spotLight.cosThetaMax = lightSample.source_param.spot.cosThetaMax;
				vert->m_desc.spotLight.cosFalloffStart = lightSample.source_param.spot.cosFalloffStart;
				vert->m_desc.spotLight.medium = lightSample.mediumIndex;
				vert->ext().init(*vert, AreaPdf::infinite(), AngularPdf::infinite(), lightSample.pChoice);
				return (int)round_to_align<8u>( this_size() + sizeof(SpotLightDesc) );
			}
			case scene::lights::LightType::AREA_LIGHT_TRIANGLE:
			case scene::lights::LightType::AREA_LIGHT_QUAD:
			case scene::lights::LightType::AREA_LIGHT_SPHERE: {
				vert->m_type = Interaction::LIGHT_AREA;
				vert->m_incidentEncoded = lightSample.source_param.area.normal;
				vert->m_desc.areaLight.intensity = lightSample.intensity;
				vert->m_desc.areaLight.medium = lightSample.mediumIndex;
			//	vert->m_desc.areaLight.primitiveId = lightSample.source_param.area.primId;
			//	vert->m_desc.areaLight.surfaceParams = lightSample.source_param.area.surfaceParams;
				vert->ext().init(*vert, AreaPdf{lightSample.pdf}, AngularPdf::infinite(), lightSample.pChoice);
				return (int)round_to_align<8u>( this_size() + sizeof(AreaLightDesc) );
			}
			case scene::lights::LightType::DIRECTIONAL_LIGHT: {
				vert->m_type = Interaction::LIGHT_DIRECTIONAL;
				vert->m_incidentEncoded = lightSample.initVec;
				vert->m_desc.dirLight.irradiance = lightSample.intensity;
				vert->m_desc.dirLight.medium = lightSample.mediumIndex;
				vert->m_desc.dirLight.projSceneArea = lightSample.source_param.dir.projSceneArea;
				vert->ext().init(*vert, AreaPdf::infinite(), AngularPdf::infinite(), lightSample.pChoice);
				return (int)round_to_align<8u>( this_size() + sizeof(DirLightDesc) );
			}
			case scene::lights::LightType::ENVMAP_LIGHT: {
				vert->m_type = Interaction::LIGHT_ENVMAP;
				vert->m_incidentEncoded = lightSample.initVec;
				vert->m_desc.dirLight.irradiance = lightSample.intensity;
				vert->m_desc.dirLight.medium = lightSample.mediumIndex;
				vert->m_desc.dirLight.projSceneArea = lightSample.source_param.dir.projSceneArea;
				vert->ext().init(*vert, AreaPdf::infinite(), AngularPdf{lightSample.pdf}, lightSample.pChoice);
				return (int)round_to_align<8u>( this_size() + sizeof(DirLightDesc) );
			}
			default: mAssert(false); break;
		}
		return 0;
	}

	inline CUDA_FUNCTION static int create_surface(void* mem, const void* previous,
		const scene::accel_struct::RayIntersectionResult& hit,
		const scene::materials::MaterialDescriptorBase& material,
		const scene::Point& position,
		const scene::TangentSpace& tangentSpace,
		const scene::Direction& incident
	) {
		//Interaction prevEventType = (previous != nullptr) ?
		//	as<PathVertex>(previous)->get_type() : Interaction::VIRTUAL;
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_position = position;
		vert->m_previous = mem != previous ? static_cast<const PathVertex*>(previous) : nullptr;
		vert->m_pathLen = previous ? static_cast<const PathVertex*>(previous)->m_pathLen + 1 : 0;
		vert->m_type = Interaction::SURFACE;
		vert->set_incident_direction(incident, hit.distance);
		vert->m_extension = ExtensionT{};
		vert->m_desc.surface.tangentSpace = tangentSpace;
		vert->m_desc.surface.primitiveId = hit.hitId;
		vert->m_desc.surface.surfaceParams = hit.surfaceParams.st;
		int size = scene::materials::fetch(material, hit.uv, &vert->m_desc.surface.material);
		return (int)round_to_align<8u>( round_to_align<8u>(this_size() + sizeof(scene::TangentSpace)
			+ sizeof(scene::PrimitiveHandle) + sizeof(scene::accel_struct::SurfaceParametrization))
			+ size);
	}

	inline CUDA_FUNCTION static int create_void(void* mem, const void* previous,
										 const scene::Direction& incident
	) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_position = scene::Point{0.0f};
		vert->m_previous = mem != previous ? static_cast<const PathVertex*>(previous) : nullptr;
		vert->m_pathLen = previous ? static_cast<const PathVertex*>(previous)->m_pathLen + 1 : 0;
		vert->m_type = Interaction::VOID;
		vert->m_incidentEncoded = incident;
		vert->m_extension = ExtensionT{};
		return (int)round_to_align<8u>( this_size() );
	}

private:
	struct AreaLightDesc {
		Spectrum intensity;
		scene::materials::MediumHandle medium;
		// TODO: area lights in the tree do not store their id -> needs more refactoring
	//	scene::PrimitiveHandle primitiveId;
	//	ei::Vec2 surfaceParams;
	};
	struct SpotLightDesc {
		ei::Vec3 intensity;
		half cosThetaMax;
		half cosFalloffStart;
		scene::materials::MediumHandle medium;
	};
	struct PointLightDesc {
		scene::materials::MediumHandle medium;
	};
	struct DirLightDesc {
		Spectrum irradiance;
		scene::materials::MediumHandle medium;
		float projSceneArea;
	};
	struct SurfaceDesc {
		scene::TangentSpace tangentSpace; // TODO: use packing?
		scene::PrimitiveHandle primitiveId;
		ei::Vec2 surfaceParams;
		scene::materials::ParameterPack material;
	};

	// The previous vertex on the path or nullptr.
	const PathVertex* m_previous;

	// The vertex position in world space. For orthographic end vertices
	// this is the direction.
	scene::Point m_position;

	// Interaction type of this vertex (the descriptor at the end of the vertex depends on this).
	Interaction m_type;
	i16 m_pathLen;

	// Direction from which this vertex was reached and its distance.
	// Stored as (x,y,len)
	// For end points the following values are stored:
	//		Cameras: view direction
	//		Point light: THE INTENSITY
	//		Spot light: center direction
	//		Area light: normal
	//		Env/Dir light: sampled light direction
	//		Void: incident direction (without encoding)
	//scene::Direction m_incident;
	ei::Vec3 m_incidentEncoded;

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
		inline CUDA_FUNCTION Desc() {}
		AreaLightDesc areaLight;
		SpotLightDesc spotLight;
		PointLightDesc pointLight;
		DirLightDesc dirLight;
		SurfaceDesc surface;
		cameras::PinholeParams pinholeCam;
		cameras::FocusParams focusCam;
	} m_desc;

	// Vertex size without the descriptor, the descriptor is 8byte aligned...
	static constexpr int this_size() {
		//return round_to_align<8u>(int(&m_desc - this));
		return (int)round_to_align<8u>( reinterpret_cast<std::ptrdiff_t>(&static_cast<PathVertex*>(nullptr)->m_desc) );
	}
};



// Interface example and dummy implementation for the vertex ExtensionT
struct VertexExtension {
	// The init function gets called at the end of vertex creation
	// for start vertices (lights and cameras).
	inline CUDA_FUNCTION void init(const PathVertex<VertexExtension>& /*thisVertex*/,
							const AreaPdf /*incidentAreaPdf*/,
							const AngularPdf /*incidentDirPdf*/,
							const float /*pChoice*/)
	{}

	// Helper to convert the parameters of the init call into a start-pdf for mis
	inline CUDA_FUNCTION static AreaPdf mis_start_pdf(const AreaPdf inAreaPdf,
								 const AngularPdf inDirPdf,
								 const float pChoice) {
		if(!inAreaPdf.is_infinite())
			return AreaPdf{ float(inAreaPdf) * pChoice };
		else if(!inDirPdf.is_infinite())
			return AreaPdf{ float(inDirPdf) * pChoice };
		else return AreaPdf{ pChoice };
	}

	// This update function gets called after the completion of a segment,
	// which is directly after the creation of the target vertex.
	// throughput: Throughput of entire subpath (all events + absorption)
	// continuationPropability: The probability of Russion roulette after
	//	sampling at prevVertex.
	inline CUDA_FUNCTION void update(const PathVertex<VertexExtension>& /*prevVertex*/,
							  const PathVertex<VertexExtension>& /*thisVertex*/,
							  const math::PdfPair /*pdf*/,
							  const Connection& /*inDir*/,
							  const Spectrum& /*throughput*/,
							  const float /*continuationPropability*/,
							  const Spectrum& /*transmission*/)
	{}

	// Helper to convert the parameters of the update call into an incident pdf for mis
	inline CUDA_FUNCTION static AreaPdf mis_pdf(const AngularPdf pdf, bool orthographic,
		const float inDistance, const float inCosAbs) {
		// Orthographic interactions get directly projected.
		if(orthographic)
			return AreaPdf{ float(pdf) * inCosAbs };
		return pdf.to_area_pdf(inCosAbs, inDistance * inDistance);
	}

	// The update function gets called whenever the excident direction changes.
	// This happens after a sampling event and after an evaluation.
	// thisVertex: Access to the vertex, for which 'this' is the payload
	//		hint: if you need information about the previous vertex use thisVertex.previous().
	// excident: The new outgoing direction (normalized)
	// pdf: Sampling PDF in both direction (forw = producing the excident vector,
	//		back = producing the incident vector)
	inline CUDA_FUNCTION void update(const PathVertex<VertexExtension>& /*thisVertex*/,
							  const scene::Direction& /*excident*/,
							  const VertexSample& /*sample*/) // TODO: ex cosine?, BRDF?
	{}
};

}} // namespace mufflon::renderer