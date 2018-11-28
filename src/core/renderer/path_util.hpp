#pragma once

#include "util/types.hpp"
#include "core/memory/dyntype_memory.hpp"
#include "core/math/sampling.hpp"
#include "core/scene/types.hpp"
#include "core/scene/materials/material_types.hpp"
#include "core/scene/lights/light_sampling.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/cameras/camera_sampling.hpp"
#include <ei/conversions.hpp>

namespace mufflon { namespace renderer {

enum class Interaction : u16 {
	VOID,				// The ray missed the scene (no intersection)
	SURFACE,			// A standard material interaction
	CAMERA_PINHOLE,		// A perspective camera start vertex
	CAMERA_ORTHO,		// An orthographic camera start vertex
	LIGHT_POINT,		// A point-light vertex
	LIGHT_DIRECTIONAL,	// A directional-light vertex (include environment map vertices)
	LIGHT_SPOT,			// A spot-light vertex
	LIGHT_AREA,			// An area-light vertex
};

struct Throughput {
	Spectrum weight;
	float guideWeight;
};

struct VertexSample {
	Spectrum throughput {0.0f};
	enum class Type: u32 {			// Type of interaction
		INVALID,
		REFLECTED,
		REFRACTED,
	} type = Type::INVALID;
	ei::Ray excident;
	AngularPdf pdfF {0.0f};			// Sampling PDF in forward direction (current sampler)
	AngularPdf pdfB {0.0f};			// Sampling PDF with reversed incident and excident directions
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
	PathVertex() : m_type(Interaction::VOID) {}

	bool is_end_point() const {
		return m_type != Interaction::SURFACE;
	}
	bool is_orthographic() const {
		return m_type == Interaction::LIGHT_DIRECTIONAL
			|| m_type == Interaction::CAMERA_ORTHO;
	}
	bool is_camera() const {
		return m_type == Interaction::CAMERA_PINHOLE || m_type == Interaction::CAMERA_ORTHO;
	}

	// Get the position of the vertex. For orthographic vertices this
	// position is computed with respect to a referencePosition.
	scene::Point get_position(const scene::Point& referencePosition) const {
		if(m_type == Interaction::LIGHT_DIRECTIONAL)
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
			case Interaction::LIGHT_POINT:
			case Interaction::LIGHT_SPOT:
				return 0.0f;
			case Interaction::LIGHT_DIRECTIONAL:
				// There is no geometrical conversion factor for hitting environment maps.
				// However, it is expected that the scene exit-distance is always the same
				// and can be used in both directions without changing the result of a
				// comparison.
				return ei::sq(scene::MAX_SCENE_SIZE);
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
	//AngularPdf get_forward_pdf() const { return m_pdfF; }
	//AngularPdf get_backward_pdf() const { return m_pdfB; }

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
					m_intensity, 0.0f,
					AngularPdf{ 1.0f / (4*ei::PI) },
					AngularPdf{ 0.0f }
				};
			}
			case Interaction::LIGHT_DIRECTIONAL: {
				const DirLightDesc* desc = as<DirLightDesc>(this->desc());
				return scene::materials::EvalValue{
					m_intensity, 0.0f,
					// Special case: the incindent area PDF is directly projected.
					// To avoid the wrong conversion later we need to do its reversal here.
					m_incidentPdf.to_angular_pdf(1.0f, scene::MAX_SCENE_SIZE * scene::MAX_SCENE_SIZE),
					AngularPdf{ 0.0f }
				};
			}
			case Interaction::LIGHT_SPOT: {
				const SpotLightDesc* desc = as<SpotLightDesc>(this->desc());
				const float cosThetaMax = __half2float(desc->cosThetaMax);
				const float cosOut = dot(desc->direction, excident);
				// Early out
				if(cosOut <= cosThetaMax) return scene::materials::EvalValue{};
				// OK, there will be some contribution
				const float cosFalloffStart = __half2float(desc->cosThetaMax);
				float falloff = scene::lights::get_falloff(cosOut, cosThetaMax, cosFalloffStart);
				return scene::materials::EvalValue{
					m_intensity * falloff, 0.0f,
					AngularPdf{ math::get_uniform_cone_pdf(cosThetaMax) },
					AngularPdf{ 0.0f }
				};
			}
			case Interaction::LIGHT_AREA: {
				const AreaLightDesc* desc = as<AreaLightDesc>(this->desc());
				const float cosOut = dot(desc->normal, excident);
				// Early out (wrong hemisphere)
				if(cosOut <= 0.0f) return scene::materials::EvalValue{};
				return scene::materials::EvalValue{
					m_intensity * cosOut, 0.0f,
					AngularPdf{ cosOut / ei::PI },
					AngularPdf{ 0.0f }
				};
			}
			case Interaction::CAMERA_PINHOLE: {
				const cameras::PinholeParams* desc = as<cameras::PinholeParams>(this->desc());
				cameras::ProjectionResult proj = pinholecam_project(*desc, excident);
				return scene::materials::EvalValue{
					Spectrum{proj.w}, 0.0f,
					proj.pdf, AngularPdf{ 0.0f }
				};
			}
		}
		return scene::materials::EvalValue{};
	}

	VertexSample sample(const math::RndSet2_1& rndSet) const {
		using namespace scene::lights;
		switch(m_type) {
			case Interaction::VOID: return VertexSample{};
			case Interaction::LIGHT_POINT: {
				auto lout = sample_light_dir_point(m_intensity, rndSet);
				return VertexSample{
					lout.flux, VertexSample::Type::REFLECTED,
					ei::Ray{m_position, lout.dir.direction},
					lout.dir.pdf, AngularPdf{0.0f},
				};
			}
			case Interaction::LIGHT_DIRECTIONAL: {
				const DirLightDesc* desc = as<DirLightDesc>(this->desc());
				return VertexSample{
					m_intensity, VertexSample::Type::REFLECTED,
					ei::Ray{m_position, desc->direction},
					desc->dirPdf, AngularPdf{0.0f}
				};
			}
			case Interaction::LIGHT_SPOT: {
				const SpotLightDesc* desc = as<SpotLightDesc>(this->desc());
				auto lout = sample_light_dir_spot(m_intensity, desc->direction, desc->cosThetaMax, desc->cosFalloffStart, rndSet);
				return VertexSample{
					lout.flux, VertexSample::Type::REFLECTED,
					ei::Ray{m_position, lout.dir.direction}, lout.dir.pdf, AngularPdf{0.0f}
				};
			}
			case Interaction::LIGHT_AREA: {
				const AreaLightDesc* desc = as<AreaLightDesc>(this->desc());
				auto lout = sample_light_dir_area(m_intensity, desc->normal, rndSet);
				return VertexSample{
					lout.flux, VertexSample::Type::REFLECTED,
					ei::Ray{m_position, lout.dir.direction}, lout.dir.pdf, AngularPdf{0.0f}
				};
			}
			case Interaction::CAMERA_PINHOLE: {
				const cameras::PinholeParams* desc = as<cameras::PinholeParams>(this->desc());
				cameras::Importon importon = pinholecam_sample_ray(*desc, m_position);
				return VertexSample{
					Spectrum{1.0f}, VertexSample::Type::REFLECTED,
					ei::Ray{m_position, importon.dir.direction},
					importon.dir.pdf, AngularPdf{0.0f}
				};
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
		const scene::accel_struct::RayIntersectionResult& intersectionResult,
		const ei::Ray& incidentRay
	) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_position = incidentRay.origin + incidentRay.direction * intersectionResult.hitT;
		vert->init_prev_offset(mem, previous);
		vert->m_type = Interaction::VOID;
		vert->m_incident = incidentRay.direction;
		vert->m_incidentPdf = AreaPdf { 0.0f };
		vert->m_extension = ExtensionT{};
		return round_to_align( sizeof(PathVertex) );
	}

	static int create_camera(void* mem, const void* previous,
		const cameras::CameraParams& camera,
		const math::PositionSample& position
	) {
		PathVertex* vert = as<PathVertex>(mem);
		vert->m_position = position.position;
		vert->init_prev_offset(mem, previous);
		//vert->m_incident = viewDir; unused
		vert->m_incidentPdf = position.pdf;
		vert->m_extension = ExtensionT{};
		if(camera.type == cameras::CameraModel::PINHOLE) {
			vert->m_type = Interaction::CAMERA_PINHOLE;
			cameras::PinholeParams* desc = as<cameras::PinholeParams>(vert->desc());
			return round_to_align( round_to_align(sizeof(PathVertex)) + sizeof(cameras::PinholeParams));
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
				desc->dirPdf = lightSample.source_param.dir.dirPdf;
				return round_to_align( round_to_align(sizeof(PathVertex)) + sizeof(DirLightDesc));
			}
			case scene::lights::LightType::ENVMAP_LIGHT: {
				vert->m_type = Interaction::LIGHT_DIRECTIONAL;
				// Environment lights are not fully sampled
				// TODO: complete sampling, REQUIRES scene BB
				DirLightDesc* desc = as<DirLightDesc>(vert->desc());
				desc->direction = lightSample.source_param.dir.direction;
				desc->dirPdf = lightSample.source_param.dir.dirPdf;
				return round_to_align( round_to_align(sizeof(PathVertex)) + sizeof(DirLightDesc));
			}
		}
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