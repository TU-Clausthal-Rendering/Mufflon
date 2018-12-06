#include "path_util.hpp"
#include "output_handler.hpp"
#include "core/memory/dyntype_memory.hpp"
#include <array>

// THIS FILE IS ONLY A PROTOTYPE OF MIS FOR TESTING THE VERTEX LAYOUT

namespace mufflon::renderer {

namespace {

// Extension which stores a partial result of the MIS-weight computation for speed-up.
struct BptVertexExt {
	// A cache to shorten the recursive evaluation of MIS.
	// It is only possible to store the second previous sum, as the previous sum
	// depends on the backward-pdf of the current vertex, which is only given in
	// the moment of the full connection.
	float m_secPrevRelativeProbabilitySum{ 0.0f };
};

using BptPathVertex = PathVertex<BptVertexExt, 4>;

// Recursive implementation for demonstration purposes
/*float get_mis_part_rec(const BptPathVertex & path0, AngularPdf pdfBack, const scene::Direction& connection, float distSq) {
	AreaPdf reversePdf = pdfBack.to_area_pdf(path0.get_geometrical_factor(connection), distSq);
	// Replace forward PDF with backward PDF (move connection one into the direction of the path-start)
	float relPdf = reversePdf / path0.get_incident_pdf();
	// Go into recursion (or not)
	const BptPathVertex* prev = path0.previous();
	if(!prev) return relPdf;	// Current one is a start vertex. There is no previous sum
	float prevRelProbabilitySum = get_mis_part_rec(*prev, path0.get_backward_pdf(), path0.get_incident_direction(), path0.get_incident_dist_sq());
	return prevRelProbabilitySum * relPdf + relPdf;
}*/

float get_mis_part(const BptPathVertex& path,
	const math::EvalValue& value,
	AngularPdf pdfBack,
	const scene::Direction& connection, float distSq,
	const void* pathMem
) {
	AreaPdf reversePdf = pdfBack.to_area_pdf(value.cosOut, distSq);
	// Replace forward PDF with backward PDF (move connection one into the direction of the path-start)
	float relPdf = reversePdf / path.get_incident_pdf();
	// Sum up all previous relative probability (cached recursion).
	// Also see PBRT p.1015.
	const BptPathVertex* prev = path.previous(pathMem);
	if(!prev) return relPdf;	// Current one is a start vertex. There is no previous sum
	// Unroll recursion further. The cache only knows the prev-prev value.
	float geoFactor = prev->get_geometrical_factor(path.get_incident_direction());
	AreaPdf prevReversePdf = value.pdfB.to_area_pdf(
		geoFactor, path.get_incident_dist_sq(pathMem));
	float relToPrev = prevReversePdf / prev->get_incident_pdf();
	return (path.ext().m_secPrevRelativeProbabilitySum * relToPrev + relToPrev) * relPdf + relPdf;
}

// Assumes that path0 and path1 are fully evaluated vertices of the end points of
// the two sub-paths.
float get_mis_weight(const BptPathVertex& path0, const math::EvalValue& value0,
					 const BptPathVertex& path1, const math::EvalValue& value1,
					 const scene::Direction& connection, float distSq,
					 const void* pathMem0, const void* pathMem1
) {
	// Compute a weight with the balance heuristic.
	// See PBRT p.1015 for details on recursive evaluation.
	float rpSum0 = get_mis_part(path0, value0, value1.pdfF, connection, distSq, pathMem0);
	float rpSum1 = get_mis_part(path1, value1, value0.pdfF, connection, distSq, pathMem1);
	// The current event has p=1 and the sum of relative probabilities (to p) is
	// given by the get_mis_part() method. => The following line is the balance
	// heuristic: how likely is this event compared to all other possibilities?
	return 1.0f / (1.0f + rpSum0 + rpSum1);
}

struct { Spectrum bxdfs; float cosines; }
connect(const BptPathVertex& path0, const BptPathVertex& path1,
		const void* pathMem0, const void* pathMem1,
		const scene::materials::Medium* media
) {
	// Some vertices will always have a contribution of 0 if connected (e.g. directional light with camera).
	if(!BptPathVertex::is_connection_possible(path0, path1)) return {Spectrum{0.0f}, 0.0f};
	// TODO: shadow test
	ei::Vec3 connection = BptPathVertex::get_connection(path0, path1);
	float distSq = lensq(connection);
	connection /= sqrt(distSq);
	auto val0 = path0.evaluate( connection, media, false);
	auto val1 = path1.evaluate(-connection, media, true);
	float mis = get_mis_weight(path0, val0, path1, val1, connection, distSq, pathMem0, pathMem1);
	return {val0.value * val1.value, val0.cosOut * val1.cosOut};
}

} // namespace ::

void foo(RenderBuffer<Device::CPU>& rb) {
	// TODO: std::array<u8, VertexFactory<>::get_size() * 2) mem;
	std::array<u8, 100> mem;
	BptPathVertex& path0tmp = *as<BptPathVertex>(mem.data());
	BptPathVertex& path1tmp = *as<BptPathVertex>(mem.data()+1);
	auto [bxdfs, cosines] = connect(path0tmp, path1tmp, mem.data(), mem.data(), nullptr);
	//rb.contribute(Pixel{0,0}, , , cosines, bxdfs);
}

} // namespace mufflon::renderer