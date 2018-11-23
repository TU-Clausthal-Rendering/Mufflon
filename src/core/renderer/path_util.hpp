#pragma once

#include "util/types.hpp"
#include "core/memory/dyntype_memory.hpp"
#include "core/math/sampling.hpp"
#include "core/scene/types.hpp"
#include "core/scene/materials/material_types.hpp"
#include "core/scene/lights/lights.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include <ei/conversions.hpp>

namespace mufflon { namespace renderer {

enum class Interaction : u16 {
	VOID,				// The ray missed the scene (no intersection)
	SURFACE,			// A standard material interaction
	CAMERA,				// A perspective camera start vertex
	CAMERA_ORTHO,		// An orthographic camera start vertex
	LIGHT_POINT,		// A point-light vertex
	LIGHT_DIRECTIONAL,	// A directional-light vertex
	LIGHT_SPOT,			// A spot-light vertex
	LIGHT_ENVMAP,		// An environment map-light vertex
	LIGHT_AREA,			// An area-light vertex
};

struct Throughput {
	Spectrum weight;
	float guideWeight;
};

// Collection of parameters produced by a random walk
// TODO: vertex customization?
struct PathHead {
	Throughput throughput;			// General throughput with guide heuristics
	scene::Point position;
	AngularPdf prevPdfF;			// Forward PDF of the last sampling PDF
	scene::Direction incident;		// May be zero-vector for start points
	AngularPdf prevPdfB;			// Backward PDF of the last sampling PDF
	Interaction type;
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
// TODO: creation factory
template < typename ExtensionT, int VERTEX_ALIGNMENT >
class PathVertex {
public:
	bool is_end_point() const {
		return m_type != Interaction::SURFACE;
	}
	bool is_orthographic() const {
		return m_type == Interaction::LIGHT_DIRECTIONAL
			|| m_type == Interaction::LIGHT_ENVMAP
			|| m_type == Interaction::CAMERA_ORTHO;
	}
	bool is_camera() const {
		return m_type == Interaction::CAMERA || m_type == Interaction::CAMERA_ORTHO;
	}

	// Get the position of the vertex. For orthographic vertices this
	// position is computed with respect to a referencePosition.
	scene::Point get_position(const scene::Point& referencePosition) const {
		if(m_type == Interaction::LIGHT_DIRECTIONAL || m_type == Interaction::LIGHT_ENVMAP)
			return referencePosition - m_position * scene::MAX_SCENE_SIZE; // Go max entities out -- should be far enough away for shadow tests
		if(m_type == Interaction::CAMERA_ORTHO)
			return referencePosition; // TODO project to near plane
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
			case Interaction::CAMERA:
			case Interaction::LIGHT_POINT:
			case Interaction::LIGHT_DIRECTIONAL:
			case Interaction::LIGHT_SPOT:
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

	// Get the sampling PDFs of this vertex (not defined, if
	// the vertex is an end point on a surface). Details at the members
	AngularPdf get_forward_pdf() const { return m_pdfF; }
	AngularPdf get_backward_pdf() const { return m_pdfB; }

	// Compute new values of the PDF and BxDF/outgoing radiance for a new
	// exident direction.
	// excident: direction pointing away from this vertex.
	// excidentDistSq: squared distance to the next vertex.
	//		This is required for the special cases of directional/environmental light sources.
	scene::materials::EvalValue evaluate(const scene::Direction & excident) const {
		switch(m_type) {
			case Interaction::VOID: return scene::materials::EvalValue{};
			case Interaction::LIGHT_POINT: {
				return scene::materials::EvalValue{
					m_incident, 0.0f,
					AngularPdf{ 1.0f / (4*ei::PI) },
					AngularPdf{ 0.0f }
				};
			}
			case Interaction::LIGHT_DIRECTIONAL: {
				return scene::materials::EvalValue{
					m_incident, 0.0f,
					// Special case: the incindent area PDF is directly projected.
					// To avoid the wrong conversion later we need to do its reversal here.
					m_incidentPdf.to_angular_pdf(1.0f, scene::MAX_SCENE_SIZE * scene::MAX_SCENE_SIZE),
					AngularPdf{ 0.0f }
				};
			}
			case Interaction::LIGHT_SPOT: {
				const SpotLightDesc* desc = as<SpotLightDesc>(this->desc());
				const float cosThetaMax = __half2float(desc->cosThetaMax);
				const float cosOut = dot(m_incident, excident);
				// Early out
				if(cosOut <= cosThetaMax) return scene::materials::EvalValue{};
				// OK, there will be some contribution
				const float cosFalloffStart = __half2float(desc->cosThetaMax);
				float falloff = scene::lights::get_falloff(cosOut, cosThetaMax, cosFalloffStart);
				return scene::materials::EvalValue{
					m_incident * falloff, 0.0f,
					AngularPdf{ math::get_uniform_cone_pdf(cosThetaMax) },
					AngularPdf{ 0.0f }
				};
			}
		}
		return scene::materials::EvalValue{};
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
	static int create_void(void* mem, const void* previous, const scene::accel_struct::RayIntersectionResult& intersectionResult, const ei::Ray& incidentRay) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_position = incidentRay.origin + incidentRay.direction * intersectionResult.hitT;
		vert->m_offsetToPath = as<u8>(previous) - as<u8>(mem);
		vert->m_type = Interaction::VOID;
		vert->m_incident = incidentRay.direction;
		vert->m_pdfF = AngularPdf { 0.0f };
		vert->m_pdfB = AngularPdf { 0.0f };
		vert->m_incidentPdf = AreaPdf { 0.0f };
		vert->m_extension = ExtensionT{};
		return round_to_align( sizeof(PathVertex) );
	}

	static int create_light(void* mem, const void* previous,
		const scene::lights::PointLight& light, AreaPdf incidentPdf, AngularPdf excidentPdf) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_position = light.position;
		vert->m_offsetToPath = as<u8>(previous) - as<u8>(mem);
		vert->m_type = Interaction::LIGHT_POINT;
		vert->m_incident = light.intensity;		// Abuse of devasted memory
		vert->m_pdfF = excidentPdf;
		vert->m_pdfB = AngularPdf { 0.0f };
		vert->m_incidentPdf = incidentPdf;
		vert->m_extension = ExtensionT{};
		return round_to_align(sizeof(PathVertex));
	}

	static int create_light(void* mem, const void* previous,
		const scene::lights::DirectionalLight& light, AreaPdf incidentPdf) {
		PathVertex* vert = as<PathVertex>(mem);
		mAssert(approx(len(light.direction), 1.0f));
		vert->m_position = light.direction;		// Abuse for special case
		vert->m_offsetToPath = as<u8>(previous) - as<u8>(mem);
		vert->m_type = Interaction::LIGHT_DIRECTIONAL;
		vert->m_incident = light.radiance;		// Abuse of devasted memory
		vert->m_pdfF = AngularPdf { 1.0f };		// Infinite in theory, use one for numerical reasons
		vert->m_pdfB = AngularPdf { 0.0f };
		vert->m_incidentPdf = incidentPdf;
		vert->m_extension = ExtensionT{};
		return round_to_align(sizeof(PathVertex));
	}

	static int create_light(void* mem, const void* previous,
		const scene::lights::SpotLight& light, AreaPdf incidentPdf, AngularPdf excidentPdf) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_position = light.position;
		vert->m_offsetToPath = as<u8>(previous) - as<u8>(mem);
		vert->m_type = Interaction::LIGHT_SPOT;
		vert->m_incident = ei::unpackOctahedral32(light.direction);		// Abuse of devasted memory
		vert->m_pdfF = excidentPdf;
		vert->m_pdfB = AngularPdf { 0.0f };
		vert->m_incidentPdf = incidentPdf;
		vert->m_extension = ExtensionT{};
		SpotLightDesc* desc = as<SpotLightDesc>(vert->desc());
		desc->intensity = light.intensity;
		desc->cosThetaMax = light.cosThetaMax;
		desc->cosFalloffStart = light.cosFalloffStart;
		return round_to_align( round_to_align(sizeof(PathVertex)) + sizeof(SpotLightDesc));
	}

private:
	struct AreaLightDesc {
		scene::Direction normal;
	};
	struct SpotLightDesc {
		ei::Vec3 intensity;
		half cosThetaMax;
		half cosFalloffStart;
	};
	struct SurfaceDesc {
		scene::TangentSpace tangentSpace;
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
	scene::Direction m_incident;

	// PDF at this vertex in forward direction.
	// Non-zero for start points and zero for end points.
	AngularPdf m_pdfF;

	// PDF in backward direction.
	// Zero for start points and non-zero for (real) end points. A path ending
	// on a surface will have a value of zero anyway.
	AngularPdf m_pdfB;

	// Area PDF of whatever created this vertex
	AreaPdf m_incidentPdf;

	// REMARK: currently 2 floats unused in 16-byte alignment
	ExtensionT m_extension;

	static constexpr int round_to_align(int s) {
		return (s + (VERTEX_ALIGNMENT-1)) & ~(VERTEX_ALIGNMENT-1);
	}

	// Private because the vertex is read-only by design (desc() is a helper for vertex creation only)
	void* desc() { return as<u8>(this) + round_to_align(sizeof(PathVertex)); }

	template<typename T, int A> 
	friend class PathVertexFactory;
};

}} // namespace mufflon::renderer