#include "cpu_bpt.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/memory/dyntype_memory.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"
#include <array>

// THIS FILE IS ONLY A PROTOTYPE OF MIS FOR TESTING THE VERTEX LAYOUT

namespace mufflon::renderer {

namespace {

//using BptPathVertex = PathVertex<struct BptVertexExt>;

// Extension which stores a partial result of the MIS-weight computation for speed-up.
struct BptVertexExt {
	//scene::Direction excident;	// The excident direction from this vertex (after sampling)
	//AngularPdf pdf;				// PDF of excident, only valid after update()
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
							const float incidentCosine, const AreaPdf incidentPdf,
							const math::Throughput& incidentThrougput) {
		this->incidentPdf = incidentPdf;
		this->throughput = incidentThrougput;
	}

	CUDA_FUNCTION void update(const BptPathVertex& thisVertex,
							  const scene::Direction& excident,
							  const AngularPdf pdfF, const AngularPdf pdfB) {
	//	this->excident = excident;
	//	this->pdf = pdfF;
		// Sum up all previous relative probability (cached recursion).
		// Also see PBRT p.1015.
		const BptPathVertex* prev = thisVertex.previous();
		if(prev) { // !prev: Current one is a start vertex. There is no previous sum
			float geoFactor = prev->get_geometrical_factor(thisVertex.get_incident_direction());
			AreaPdf prevReversePdf = pdfB.to_area_pdf(
				geoFactor, thisVertex.get_incident_dist_sq());
			// Replace forward PDF with backward PDF (move connection one into the direction of the path-start)
			float relToPrev = prevReversePdf / prev->ext().incidentPdf;
			prevRelativeProbabilitySum = relToPrev + relToPrev * prev->ext().prevRelativeProbabilitySum;
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

float get_mis_part(const BptVertexExt& pathExt,
	const math::EvalValue& value,
	AngularPdf pdfBack,
	float distSq
) {
	AreaPdf reversePdf = pdfBack.to_area_pdf(value.cosOut, distSq);
	// Replace forward PDF with backward PDF (move connection one into the direction of the path-start)
	float relPdf = reversePdf / pathExt.incidentPdf;
	// Sum up all previous relative probability (cached recursion).
	// Also see PBRT p.1015.
//	const BptPathVertex* prev = path.previous();
//	if(!prev) return relPdf;	// Current one is a start vertex. There is no previous sum
	return pathExt.prevRelativeProbabilitySum * relPdf + relPdf; 
}

// Assumes that path0 and path1 are fully evaluated vertices of the end points of
// the two sub-paths.
float get_mis_weight(const BptVertexExt& path0Ext, const math::EvalValue& value0,
					 const BptVertexExt& path1Ext, const math::EvalValue& value1,
					 float distSq
) {
	// Compute a weight with the balance heuristic.
	// See PBRT p.1015 for details on recursive evaluation.
	float rpSum0 = get_mis_part(path0Ext, value0, value1.pdfF, distSq);
	float rpSum1 = get_mis_part(path1Ext, value1, value0.pdfF, distSq);
	// The current event has p=1 and the sum of relative probabilities (to p) is
	// given by the get_mis_part() method. => The following line is the balance
	// heuristic: how likely is this event compared to all other possibilities?
	return 1.0f / (1.0f + rpSum0 + rpSum1);
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
	Spectrum bxdfProd = val0.value * val1.value;
	float cosProd = path0.is_surface() ? val0.cosOut : 1.0f;//TODO: abs?
	if(path1.is_surface()) cosProd *= val1.cosOut;
	mAssert(cosProd >= 0.0f);
	// Early out if there would not be a contribution (estimating the materials is usually
	// cheaper than the any-hit test).
	if(any(greater(bxdfProd, 0.0f)) && cosProd > 0.0f) {
		// Shadow test
		if(!scene::accel_struct::any_intersection(
				scene, { connection.v0, connection.dir },
				path0.get_primitive_id(), connection.distance)) {
			auto ext0Copy = path0.ext();
			auto ext1Copy = path1.ext();
			ext0Copy.update(path0, connection.dir, val0.pdfF, val0.pdfB);
			ext1Copy.update(path1, connection.dir, val1.pdfF, val1.pdfB);
			float mis = get_mis_weight(ext0Copy, val0, ext1Copy, val1, connection.distanceSq);
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
	scene::lights::Photon p = emit(m_sceneDesc.lightTree, idx, outputBuffer.get_num_pixels(),
		m_rngs[idx].next(), m_sceneDesc.aabb, rndStart);
	BptPathVertex::create_light(&path[0], &path[0], p, m_rngs[idx]);
	math::Throughput throughput;
	VertexSample sample;

	int lightPathLen = 0;
	do {
		// Walk
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		if(!walk(m_sceneDesc, path[lightPathLen], rnd, -1.0f, true, throughput, path[lightPathLen+1], sample))
			break;
		++lightPathLen;
	} while(lightPathLen < m_params.maxPathLength-1); // -1 because there is at least one segment on the view path

	// Trace view path
	BptPathVertex vertex[2];
	// Create a start for the path
	BptPathVertex::create_camera(&vertex[0], &vertex[0], m_sceneDesc.camera.get(), coord, m_rngs[idx].next());
	int currentV = 0;
	throughput = math::Throughput{};
	int viewPathLen = 0;
	do {
		// Make a connection to any event on the light path
		int maxL = ei::min(lightPathLen+1, m_params.maxPathLength-viewPathLen);
		for(int l = ei::max(0, m_params.minPathLength-viewPathLen-1); l < maxL; ++l) {
			Pixel outCoord = coord;
			auto conVal = connect(vertex[currentV], path[l], m_sceneDesc, outCoord);
			outputBuffer.contribute(outCoord, throughput, path[l].ext().throughput, conVal.cosines, conVal.bxdfs);
		}

		// Walk
		int otherV = 1 - currentV;
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		if(!walk(m_sceneDesc, vertex[currentV], rnd, -1.0f, true, throughput, vertex[otherV], sample)) {
			if(throughput.weight != Spectrum{ 0.f }) {
				// Missed scene - sample background
				auto background = evaluate_background(m_sceneDesc.lightTree.background, sample.excident);
				if(any(greater(background.value, 0.0f))) {
					float relPdf = background.pdfB / sample.pdfF;
					float mis = 1.0f / (1.0f + vertex[currentV].ext().prevRelativeProbabilitySum * relPdf + relPdf);
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
				vertex[currentV].ext().update(vertex[currentV], scene::Direction{0.0f}, AngularPdf{0.0f}, emission.pdf);
				AreaPdf startPdf = emit_pdf(m_sceneDesc.lightTree, vertex[currentV].get_primitive_id(),
											vertex[currentV].get_surface_params(), vertex[1-currentV].get_position(),
											scene::lights::guide_flux);
				float relPdf = startPdf / vertex[currentV].ext().incidentPdf;
				float mis = 1.0f / (1.0f + vertex[currentV].ext().prevRelativeProbabilitySum * relPdf + relPdf);
				emission.value *= mis;
			}
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
