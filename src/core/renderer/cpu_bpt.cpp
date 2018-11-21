#include "path_util.hpp"
#include "output_handler.hpp"
#include "core/memory/dyntype_memory.hpp"
#include <array>

// THIS FILE IS ONLY A PROTOTYPE OF MIS FOR TESTING THE VERTEX LAYOUT

namespace mufflon::renderer {

namespace {

// Extension which stores a partial result of the MIS-weight computation for speed-up.
struct BptVertexExt {
	float m_prevRelativeProbabilitySum;
};

using BptPathVertex = PathVertex<BptVertexExt>;

float get_mis_part(const BptPathVertex & path0, AngularPdf pdfBack, const scene::Direction& connection, float distSq) {
	AreaPdf reversePdf = pdfBack.to_area_pdf(path0.get_geometrical_factor(connection), distSq);
	// Replace forward PDF with backward PDF (move connection one into the direction of the path-start)
	float relPdf = reversePdf / path0.get_incident_pdf();
	// Sum up all previous relative probability (cached recursion).
	// The cache is stored relative to this vertex and not to the newley connected one.
	// Therefore, it must be multiplied with the local relativity.
	// Also see PBRT p.1015.
	float relToPrev = 1.0f; // TODO: needs correct backward pdf of path0
	// TODO / path0.previous()->get_incident_pdf()
	return path0.ext().m_prevRelativeProbabilitySum * relToPrev * relPdf + relPdf;
}

// Assumes that path0 and path1 are fully evaluated vertices of the end points of
// the two sub-paths.
float get_mis_weight(const BptPathVertex& path0,
					 const BptPathVertex& path1,
					 const scene::Direction& connection, float distSq
) {
	// Compute a weight with the balance heuristic.
	// See PBRT p.1015 for details on recursive evaluation.
	float rpSum0 = get_mis_part(path0, path1.get_forward_pdf(), connection, distSq);
	float rpSum1 = get_mis_part(path1, path0.get_forward_pdf(), connection, distSq);
	// The current event has p=1 and the sum of relative probabilities (to p) is
	// given by the get_mis_part() method. => The following line is the balance
	// heuristic: how likely is this event compared to all other possibilities?
	return 1.0f / (1.0f + rpSum0 + rpSum1);
}

struct { ei::Vec3 bxdfs; float cosines; }
connect(const BptPathVertex& path0, const BptPathVertex& path1) {
	ei::Vec3 connection = path0.get_position() - path1.get_position();
	float distSq = lensq(connection);
	connection /= sqrt(distSq);
	auto val0 = path0.evaluate(-connection);
	auto val1 = path1.evaluate( connection);
	// Evaluate did not change the original vertices. Instead we create two new
	// temporary ones.
	// TODO: std::array<u8, VertexFactory<>::get_size() * 2) mem;
	std::array<u8, 100> mem;
	BptPathVertex& path0tmp = *as<BptPathVertex>(mem.data());
	BptPathVertex& path1tmp = *as<BptPathVertex>(mem.data()+1);
	float mis = get_mis_weight(path0tmp, path1tmp, connection, distSq);
	return {val0.bxdf * val1.bxdf, val0.cosThetaOut * val1.cosThetaOut};
}

} // namespace ::

void foo(RenderBuffer<Device::CPU>& rb) {
	// TODO: std::array<u8, VertexFactory<>::get_size() * 2) mem;
	std::array<u8, 100> mem;
	BptPathVertex& path0tmp = *as<BptPathVertex>(mem.data());
	BptPathVertex& path1tmp = *as<BptPathVertex>(mem.data()+1);
	auto [bxdfs, cosines] = connect(path0tmp, path1tmp);
	//rb.contribute(Pixel{0,0}, , , cosines, bxdfs);
}

} // namespace mufflon::renderer