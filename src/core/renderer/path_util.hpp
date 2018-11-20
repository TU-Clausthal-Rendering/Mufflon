#pragma once

#include "util/types.hpp"
#include "core/memory/dyntype_memory.hpp"
#include "core/scene/types.hpp"
#include "core/scene/materials/types.hpp"

namespace mufflon { namespace renderer {

enum class Interaction : u16 {
	VOID,				// The ray missed the scene (no intersection)
	SURFACE,			// A standard material interaction
	CAMERA,				// A camera start vertex
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
	float prevPdfF;					// Forward PDF of the last sampling PDF
	scene::Direction incident;		// May be zero-vector for start points
	float prevPdfB;					// Backward PDF of the last sampling PDF
	// TODO: prevAreaPDF ?
	Interaction type;
};

/*
 * The vertex is an abstraction layer between light/camera/material implementation
 * and a general rendering routine.
 * The type is very versatile as its size varyes for each different Interaction and Renderer.
 * A vertex is: a header (fixed size), renderer specific payload (template arguments) and
 * the descriptor for the interaction (size depends on interaction).
 */
// TODO: template payload
// TODO: creation factory
// TODO: update function(s)
class Vertex {
public:
	scene::Point get_position() const { return m_position; }
	scene::Direction get_incident_direction() const { return m_incident; }

	// The per area PDF of the previous segment which generated this vertex
	AreaPdf get_incident_pdf() const { return m_incidentPdf; }
	// Get the per area PDF, if this vertex is approached from a different vertex.
	// prevForwardPdf: previous sampling pdf which gets converted to the AreaPdf
	//		dependent on the current vertex type.
	// connection: Connection direction (normalized) from/to the source vertex. The
	//		sign of the direction can be arbitrary.
	AreaPdf get_incident_pdf(AngularPdf prevForwardPdf, const scene::Direction & connection, float distSq) const {
		switch(m_type) {
			// Non-hitable vertices
			case Interaction::VOID:
			case Interaction::CAMERA:
			case Interaction::LIGHT_POINT:
			case Interaction::LIGHT_DIRECTIONAL:
			case Interaction::LIGHT_SPOT:
				return AreaPdf{ 0.0f };
			case Interaction::LIGHT_ENVMAP:
				// There is no geometrical conversion factor for hitting environment maps.
				// However, it is expected that the scene exit-distance is always the same
				// and can be used in both directions without changing the result of a
				// comparison.
				return prevForwardPdf.to_area_pdf(1.0f, distSq);
			case Interaction::LIGHT_AREA: {
				auto* alDesc = as<AreaLightDesc>(this + 1); // Descriptor at end of vertex
				float iDotN = dot(connection, alDesc->normal);
				return prevForwardPdf.to_area_pdf(iDotN, distSq);
			}
			case Interaction::SURFACE: {
				auto* surfDesc = as<SurfaceDesc>(this + 1); // Descriptor at end of vertex
				float iDotN = dot(connection, surfDesc->tangentSpace.shadingN);
				return prevForwardPdf.to_area_pdf(iDotN, distSq);
			}
		}
		return AreaPdf{ 0.0f };
	}

	// Get the sampling PDFs of this vertex (not defined, if
	// the vertex is an end point on a surface). Details at the members
	AngularPdf get_forward_pdf() const { return m_pdfF; }
	AngularPdf get_backward_pdf() const { return m_pdfB; }

	// Compute new values of the PDF and BxDF/outgoing radiance for a new
	// exident direction.
	// excident: direction pointing away from this vertex.
	scene::materials::EvalValue evaluate(const scene::Direction & excident) const {
		// TODO
		return scene::materials::EvalValue{};
	}
protected:
	struct AreaLightDesc {
		scene::Direction normal;
	};
	struct SurfaceDesc {
		scene::TangentSpace tangentSpace;
	};


	// The vertex  position in world space
	scene::Point m_position;

	// Byte offset to the previous vertex (can also be used as an index,
	// if vertices are packed with constant space).
	// Intended use in packed buffers: prev = as<Vertex>(as<u8>(this) + m_offsetToPrevious)
	short m_offsetToPrevious;

	// Interaction type of this vertex (the descriptor at the end of the vertex depends on this).
	Interaction m_type;

	// Direction from which this vertex was reached.
	// May be zero-vector for start points.
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

	// REMARK: currently 2 floats of unused in 16-byte alignment
};

}} // namespace mufflon::renderer