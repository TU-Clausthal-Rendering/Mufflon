#include "cpu_bpt.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/memory/dyntype_memory.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include <array>


namespace mufflon::renderer {

namespace {

// Extension which stores a partial result of the MIS-weight computation for speed-up.
struct BptVertexExt {
	AreaPdf incidentPdf;
	// A cache to shorten the recursive evaluation of MIS.
	// It is only possible to store the previous sum, as the current sum
	// depends on the backward-pdf of the next vertex, which is only given in
	// the moment of the full connection.
	// only valid after update().
	float prevRelativeProbabilitySum{ 0.0f };

	math::Throughput throughput;	// Throughput of the path up to this point

	CUDA_FUNCTION void init(const BptPathVertex& thisVertex,
							const scene::Direction& incident, const float incidentDistance,
							const AreaPdf incidentPdf, const float incidentCosineAbs,
							const math::Throughput& incidentThrougput) {
		this->incidentPdf = incidentPdf;
		this->throughput = incidentThrougput;
	}

	CUDA_FUNCTION void update(const BptPathVertex& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair pdf) {
		// Sum up all previous relative probability (cached recursion).
		// Also see PBRT p.1015.
		const BptPathVertex* prev = thisVertex.previous();
		if(prev) { // !prev: Current one is a start vertex. There is no previous sum
			AreaPdf prevReversePdf = prev->convert_pdf(thisVertex.get_type(), pdf.back,
				{thisVertex.get_incident_direction(), thisVertex.get_incident_dist_sq()}).pdf;
			// Replace forward PDF with backward PDF (move connection one into the direction of the path-start)
			float relToPrev = prevReversePdf / prev->ext().incidentPdf;
			prevRelativeProbabilitySum = relToPrev + relToPrev * prev->ext().prevRelativeProbabilitySum;
		}
	}
};


// Get the sum of relative sampling probabilities for all events (connections/random hit)
// up to the current vertex.
// vertex: The vertex for which we want to compute the sum of previous sampling events.
// vertexPdfs: Newly evaluated pdfs for the current connection.
// pdfBack: The backward pdf of the next event.
// connection: Direction and distance needed to convert pdfs.
float get_mis_part(const BptPathVertex& vertex, const AngularPdf& vertexPdfBack,
				   AngularPdf nextPdfBack, Interaction nextEventType,
				   const ConnectionDir& connection) {
	// 1. Prev connection probability: got to the current vertex from the next one instead
	// of from the previous one.
	AreaPdf reversePdf = vertex.convert_pdf(nextEventType, nextPdfBack, connection).pdf;
	float relPdf = reversePdf / vertex.ext().incidentPdf;
	// 2. Prev-prev connection probability: only now that we have a connection we
	// know 'vertexPdfs' which is necessary to do the same computation at the previous vertex.
	const BptPathVertex* prev = vertex.previous();
	if(prev) {
		reversePdf = prev->convert_pdf(vertex.get_type(), vertexPdfBack,
			{ vertex.get_incident_direction(), vertex.get_incident_dist_sq() }).pdf;
		float prevRelPdf = reversePdf / prev->ext().incidentPdf;
		// For older events we have a result in ext().prevRelativeProbabilitySum
		// (cached recursion). Also see PBRT p.1015.
		return relPdf + relPdf * (prevRelPdf + prevRelPdf * prev->ext().prevRelativeProbabilitySum);
	} else
		return relPdf;
}

// Get the final mis weight for a connection between two vertices.
float get_mis_weight(const BptPathVertex& vertex0, math::PdfPair pdf0,
					 const BptPathVertex& vertex1, math::PdfPair pdf1,
					 const ConnectionDir& connection
) {
	// Compute a weight with the balance heuristic.
	// See PBRT p.1015 for details on recursive evaluation.
	float rpSum0 = get_mis_part(vertex0, pdf0.back, pdf1.forw, vertex1.get_type(), connection);
	float rpSum1 = get_mis_part(vertex1, pdf1.back, pdf0.forw, vertex0.get_type(), connection);
	// The current event has p=1 and the sum of relative probabilities (to p) is
	// given by the get_mis_part() method. => The following line is the balance
	// heuristic: how likely is this event compared to all other possibilities?
	float weight = 1.0f / (1.0f + rpSum0 + rpSum1);
	mAssert(!isnan(weight));
	return weight;
}

// Get the final mis weight for a random hit
float get_mis_weight(const BptPathVertex& vertex,
					 AngularPdf vertexPdfBack,
					 AreaPdf startPdf
) {
	// Compute a weight with the balance heuristic.
	// See PBRT p.1015 for details on recursive evaluation.
	float prevRelativeProbabilitySum = get_mis_part(vertex, vertexPdfBack, AngularPdf{float(startPdf)},
		Interaction::VIRTUAL, {scene::Direction{0.0f}, 0.0f});
	float weight = 1.0f / (1.0f + prevRelativeProbabilitySum);
	mAssert(!isnan(weight));
	return weight;
}

struct ConnectionValue { Spectrum bxdfs; float cosines; };
ConnectionValue connect(const BptPathVertex& path0, const BptPathVertex& path1,
						const scene::SceneDescriptor<Device::CPU>& scene,
						Pixel& coord
) {
	// Some vertices will always have a contribution of 0 if connected (e.g. directional light with camera).
	if(!BptPathVertex::is_connection_possible(path0, path1)) return {Spectrum{0.0f}, 0.0f};
	Connection connection = BptPathVertex::get_connection(path0, path1);
	auto val0 = path0.evaluate( connection.dir, scene.media, coord, false);
	auto val1 = path1.evaluate(-connection.dir, scene.media, coord, true);
	// Cancel reprojections outside the screen
	if(coord.x == -1) return {Spectrum{0.0f}, 0.0f};
	Spectrum bxdfProd = val0.value * val1.value;
	float cosProd = val0.cosOut * val1.cosOut;//TODO: abs?
	mAssert(cosProd >= 0.0f);
	mAssert(!isnan(bxdfProd.x));
	// Early out if there would not be a contribution (estimating the materials is usually
	// cheaper than the any-hit test).
	if(any(greater(bxdfProd, 0.0f)) && cosProd > 0.0f) {
		// Shadow test
		if(!scene::accel_struct::any_intersection(
				scene, { connection.v0, connection.dir },
				path0.get_geometric_normal(), connection.distance)) {
			float mis = get_mis_weight(path0, val0.pdf, path1, val1.pdf, connection);
			return {bxdfProd * (mis / connection.distanceSq), cosProd};
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
	std::vector<std::vector<BptPathVertex>> pathMem(get_thread_num());
	for(auto& it : pathMem) it.resize(m_params.maxPathLength);

#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
		this->sample(Pixel{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() },
			pixel, m_outputBuffer, pathMem[get_current_thread_idx()]);
	}

	Profiler::instance().create_snapshot_all();
}

void CpuBidirPathTracer::on_reset() {
	init_rngs(m_outputBuffer.get_num_pixels());
}

void CpuBidirPathTracer::sample(const Pixel coord, int idx,
								RenderBuffer<Device::CPU>& outputBuffer,
								std::vector<BptPathVertex>& path) {
	// Trace a light path
	math::RndSet2_1 rndStart { m_rngs[idx].next(), m_rngs[idx].next() };
	u64 lightTreeSeed = m_rngs[idx].next();
	scene::lights::Photon p = emit(m_sceneDesc.lightTree, idx, outputBuffer.get_num_pixels(),
		lightTreeSeed, m_sceneDesc.aabb, rndStart);
	BptPathVertex::create_light(&path[0], nullptr, p, m_rngs[idx]);
	math::Throughput throughput;
	VertexSample sample;

	int lightPathLen = 0;
	if(m_params.maxPathLength > 1) do {
		// Walk
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		if(!walk(m_sceneDesc, path[lightPathLen], rnd, rndRoulette, true, throughput, path[lightPathLen+1], sample))
			break;
		++lightPathLen;
	} while(lightPathLen < m_params.maxPathLength-1); // -1 because there is at least one segment on the view path

	// Trace view path
	BptPathVertex vertex[2];
	// Create a start for the path
	BptPathVertex::create_camera(&vertex[0], nullptr, m_sceneDesc.camera.get(), coord, m_rngs[idx].next());
	int currentV = 0;
	throughput = math::Throughput{};
	int viewPathLen = 0;
	do {
		// Make a connection to any event on the light path
		int maxL = ei::min(lightPathLen+1, m_params.maxPathLength-viewPathLen);
		for(int l = ei::max(0, m_params.minPathLength-viewPathLen-1); l < maxL; ++l) {
			Pixel outCoord = coord;
			auto conVal = connect(vertex[currentV], path[l], m_sceneDesc, outCoord);
			mAssert(!isnan(conVal.cosines) && !isnan(conVal.bxdfs.x) && !isnan(throughput.weight.x) && !isnan(path[l].ext().throughput.weight.x));
			outputBuffer.contribute(outCoord, throughput, path[l].ext().throughput, conVal.cosines, conVal.bxdfs);
		}

		// Walk
		int otherV = 1 - currentV;
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		if(!walk(m_sceneDesc, vertex[currentV], rnd, rndRoulette, false, throughput, vertex[otherV], sample)) {
			if(throughput.weight != Spectrum{ 0.0f }) {
				// Missed scene - sample background
				auto background = evaluate_background(m_sceneDesc.lightTree.background, sample.excident);
				if(any(greater(background.value, 0.0f))) {
					// Update MIS for the last connection and that before.
					// For direction/envmap sources the sampling of position and direction is
					// reverted, so we need to cast the swapped pdfs to fit the expected order of events.
					AngularPdf backtracePdf = AngularPdf{ 1.0f / math::projected_area(sample.excident, m_sceneDesc.aabb) };
					AreaPdf startPdf = background_pdf(m_sceneDesc.lightTree, background);
					float mis = get_mis_weight(vertex[otherV], backtracePdf, startPdf);
					background.value *= mis;
					m_outputBuffer.contribute(coord, throughput, background.value,
											ei::Vec3{ 0, 0, 0 }, ei::Vec3{ 0, 0, 0 },
											ei::Vec3{ 0, 0, 0 });
				}
			}
			break;
		}
		++viewPathLen;
		currentV = otherV;

		// Evaluate direct hit of area ligths
		if(viewPathLen >= m_params.minPathLength) {
			math::SampleValue emission = vertex[currentV].get_emission();
			if(emission.value != 0.0f) {
				AreaPdf startPdf = emit_pdf(m_sceneDesc.lightTree, vertex[currentV].get_primitive_id(),
											vertex[currentV].get_surface_params(), vertex[1-currentV].get_position(),
											scene::lights::guide_flux);
				float mis = get_mis_weight(vertex[currentV], emission.pdf, startPdf);
				emission.value *= mis;
			}
			mAssert(!isnan(emission.value.x));
			m_outputBuffer.contribute(coord, throughput, emission.value, vertex[currentV].get_position(),
									vertex[currentV].get_normal(), vertex[currentV].get_albedo());
		}
	} while(viewPathLen < m_params.maxPathLength);
}

void CpuBidirPathTracer::init_rngs(int num) {
	m_rngs.resize(num);
	// TODO: incude some global seed into the initialization
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i);
}

} // namespace mufflon::renderer
