#include "cpu_bpt.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/memory/dyntype_memory.hpp"
#include <array>

// THIS FILE IS ONLY A PROTOTYPE OF MIS FOR TESTING THE VERTEX LAYOUT

namespace mufflon::renderer {

namespace {

using BptPathVertex = PathVertex<struct BptVertexExt, 4>;

// Extension which stores a partial result of the MIS-weight computation for speed-up.
struct BptVertexExt {
	scene::Direction excident;	// The excident direction from this vertex (after sampling)
	AngularPdf pdf;				// PDF of excident, only valid after update()
	AreaPdf incidentPdf;
	// A cache to shorten the recursive evaluation of MIS.
	// It is only possible to store the previous sum, as the current sum
	// depends on the backward-pdf of the next vertex, which is only given in
	// the moment of the full connection.
	// only valid after update().
	float prevRelativeProbabilitySum{ 0.0f };

	CUDA_FUNCTION void init(const PathVertex<BptVertexExt, 4>& thisVertex,
			  const scene::Direction& incident, const float incidentDistance,
			  const float incidentCosine, const AreaPdf incidentPdf) {
		this->incidentPdf = incidentPdf;
	}

	CUDA_FUNCTION void update(const PathVertex<BptVertexExt, 4>& thisVertex,
				const math::PathSample& sample) {
		excident = sample.excident;
		pdf = sample.pdfF;
		// Sum up all previous relative probability (cached recursion).
		// Also see PBRT p.1015.
		const BptPathVertex* prev = thisVertex.previous();
		if(prev) { // !prev: Current one is a start vertex. There is no previous sum
			float geoFactor = prev->get_geometrical_factor(thisVertex.get_incident_direction());
			AreaPdf prevReversePdf = sample.pdfB.to_area_pdf(
				geoFactor, thisVertex.get_incident_dist_sq());
			// Replace forward PDF with backward PDF (move connection one into the direction of the path-start)
			float relToPrev = prevReversePdf / prev->ext().incidentPdf;
			prevRelativeProbabilitySum = relToPrev + relToPrev * prev->ext().prevRelativeProbabilitySum;
			// TODO: LT reuse special case
		}
	}
};

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
	float relPdf = reversePdf / path.ext().incidentPdf;
	// Sum up all previous relative probability (cached recursion).
	// Also see PBRT p.1015.
	const BptPathVertex* prev = path.previous();
	if(!prev) return relPdf;	// Current one is a start vertex. There is no previous sum
	return path.ext().prevRelativeProbabilitySum * relPdf + relPdf; 
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

struct ConnectionValue { Spectrum bxdfs; float cosines; };

ConnectionValue connect(const BptPathVertex& path0, const BptPathVertex& path1,
						const void* pathMem0, const void* pathMem1,
						const scene::SceneDescriptor<Device::CPU>& scene
) {
	// Some vertices will always have a contribution of 0 if connected (e.g. directional light with camera).
	if(!BptPathVertex::is_connection_possible(path0, path1)) return {Spectrum{0.0f}, 0.0f};
	Connection connection = BptPathVertex::get_connection(path0, path1);
	auto val0 = path0.evaluate( connection.dir, scene.media, false);
	auto val1 = path1.evaluate(-connection.dir, scene.media, true);
	Spectrum bxdfProd = val0.value * val1.value;
	// Early out if there would not be a contribution (estimating the materials is usually
	// cheaper than the any-hit test).
	if(any(greater(bxdfProd, 0.0f))) {
		// Shadow test
		if(scene::accel_struct::any_intersection_scene_lbvh<Device::CPU>(
				scene, { connection.v0, connection.dir },
				path0.get_primitive_id(), connection.distance)) {
			float mis = get_mis_weight(path0, val0, path1, val1, connection.dir, connection.distanceSq, pathMem0, pathMem1);
			return {val0.value * val1.value, val0.cosOut * val1.cosOut};
		}
	}
	return {Spectrum{0.0f}, 0.0f};
}

} // namespace ::


CpuBidirPathTracer::CpuBidirPathTracer() {
	// The BPT does not need additional memory resources like photon maps.
}

void CpuBidirPathTracer::iterate() {
	auto scope = Profiler::instance().start<CpuProfileState>("CPU BPT iteration", ProfileLevel::HIGH);

	// Allocate a path memory (up to pathlength many vertices per thread)
	std::vector<std::vector<BptPathVertex>> pathMem;

#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
		this->sample(Pixel{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() }, m_outputBuffer);
	}

	Profiler::instance().create_snapshot_all();
}

void CpuBidirPathTracer::on_reset() {
	init_rngs(m_outputBuffer.get_num_pixels());
}

void CpuBidirPathTracer::sample(const Pixel coord, RenderBuffer<Device::CPU>& outputBuffer) {
	// Trace a light path

	// Trace view path
}

void CpuBidirPathTracer::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

void foo(RenderBuffer<Device::CPU>& rb) {
	// TODO: std::array<u8, VertexFactory<>::get_size() * 2) mem;
/*	std::array<u8, 100> mem;
	BptPathVertex& path0tmp = *as<BptPathVertex>(mem.data());
	BptPathVertex& path1tmp = *as<BptPathVertex>(mem.data()+1);
	auto [bxdfs, cosines] = connect(path0tmp, path1tmp, mem.data(), mem.data(), nullptr);*/
	//rb.contribute(Pixel{0,0}, , , cosines, bxdfs);
}

} // namespace mufflon::renderer