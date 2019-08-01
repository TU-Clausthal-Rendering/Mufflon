#include "cpu_ivcm.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/cameras/camera.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/output_handler.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"

#include <cn/rnd.hpp>


#include "core/math/curvature.hpp"

namespace mufflon { namespace scene { namespace accel_struct {
	CUDA_FUNCTION float compute_face_curvature(
		const SceneDescriptor<CURRENT_DEV>& scene,
		const PrimitiveHandle& hitId,
		const Direction& geoNormal) {
	const LodDescriptor<CURRENT_DEV>& object = scene.lods[scene.lodIndices[hitId.instanceId]];
	const u32* vertexIndices = object.polygon.vertexIndices;
	const Direction* normals = object.polygon.normals;
	const Point* positions = object.polygon.vertices;
	if(static_cast<u32>(hitId.primId) < object.polygon.numTriangles) {
		// Triangle
		vertexIndices += 3u * hitId.primId;
		const ei::Mat3x3 transform { scene.instanceToWorld[hitId.instanceId] };
		const ei::Mat3x3 rotationInvScale = transpose(ei::Mat3x3{ scene.worldToInstance[hitId.instanceId] });
		/*return math::compute_mean_curvature(transform * positions[vertexIndices[0]],
											 transform * positions[vertexIndices[1]],
											 transform * positions[vertexIndices[2]],
											 normalize(rotationInvScale * normals[vertexIndices[0]]),
											 normalize(rotationInvScale * normals[vertexIndices[1]]),
											 normalize(rotationInvScale * normals[vertexIndices[2]]));*/
		auto c = math::compute_curvature(geoNormal, transform * positions[vertexIndices[0]],
											 transform * positions[vertexIndices[1]],
											 transform * positions[vertexIndices[2]],
											 normalize(rotationInvScale * normals[vertexIndices[0]]),
											 normalize(rotationInvScale * normals[vertexIndices[1]]),
											 normalize(rotationInvScale * normals[vertexIndices[2]]));
		return c.get_mean_curvature();
	} if(static_cast<u32>(hitId.primId) < object.polygon.numTriangles + object.polygon.numQuads) {
		// Quad
		const u32 quadId = (hitId.primId - object.polygon.numTriangles);
		vertexIndices += 4u * quadId + 3u * object.polygon.numTriangles;
		const ei::Mat3x3 transform { scene.instanceToWorld[hitId.instanceId] };
		const ei::Mat3x3 rotationInvScale = transpose(ei::Mat3x3{ scene.worldToInstance[hitId.instanceId] });
		/*return math::compute_mean_curvature(transform * positions[vertexIndices[0]],
											 transform * positions[vertexIndices[1]],
											 transform * positions[vertexIndices[2]],
											 transform * positions[vertexIndices[3]],
											 normalize(rotationInvScale * normals[vertexIndices[0]]),
											 normalize(rotationInvScale * normals[vertexIndices[1]]),
											 normalize(rotationInvScale * normals[vertexIndices[2]]),
											 normalize(rotationInvScale * normals[vertexIndices[3]]));*/
		auto c = math::compute_curvature(geoNormal, transform * positions[vertexIndices[0]],
											 transform * positions[vertexIndices[1]],
											 transform * positions[vertexIndices[2]],
											 transform * positions[vertexIndices[3]],
											 normalize(rotationInvScale * normals[vertexIndices[0]]),
											 normalize(rotationInvScale * normals[vertexIndices[1]]),
											 normalize(rotationInvScale * normals[vertexIndices[2]]),
											 normalize(rotationInvScale * normals[vertexIndices[3]]));
		return c.get_mean_curvature();
	} else {
		// Sphere
		// Assume uniform scaling from instancing:
		float scale = len(ei::Vec3{scene.worldToInstance[hitId.instanceId]});
		return scale / object.spheres.spheres[hitId.primId].radius;
	}
}}}} // namespace mufflon::scene::accel_struct


namespace mufflon::renderer {

namespace {

// Extension which stores a partial result of the MIS-weight computation for speed-up.
struct IvcmVertexExt {
	AreaPdf incidentPdf;
	Spectrum throughput;
	AngularPdf pdfBack;
	// Store 'cosθ / d²' for the previous vertex OR 'cosθ / (d² samplePdf n A)' for hitable light sources
	float prevConversionFactor { 0.0f };
	float density;
	Footprint2D footprint;


	CUDA_FUNCTION void init(const IvcmPathVertex& thisVertex,
							const AreaPdf inAreaPdf,
							const AngularPdf inDirPdf,
							const float pChoice) {
		this->incidentPdf = VertexExtension::mis_start_pdf(inAreaPdf, inDirPdf, pChoice);
		this->throughput = Spectrum{1.0f};
		this->footprint.init(1.0f / float(inAreaPdf), 1.0f / float(inDirPdf));
	}

	CUDA_FUNCTION void update(const IvcmPathVertex& prevVertex,
							  const IvcmPathVertex& thisVertex,
							  const math::PdfPair pdf,
							  const Connection& incident,
							  const math::Throughput& throughput) {
		float inCosAbs = ei::abs(thisVertex.get_geometric_factor(incident.dir));
		float outCosAbs = ei::abs(prevVertex.get_geometric_factor(incident.dir));
		bool orthoConnection = prevVertex.is_orthographic() || thisVertex.is_orthographic();
		this->incidentPdf = VertexExtension::mis_pdf(pdf.forw, orthoConnection, incident.distance, inCosAbs);
		this->throughput = throughput.weight;
		if(prevVertex.is_hitable()) {
			// Compute as much as possible from the conversion factor.
			// At this point we do not know n and A for the photons. This quantities
			// are added in the kernel after the walk.
			this->prevConversionFactor = orthoConnection ? outCosAbs : outCosAbs / incident.distanceSq;
		}
		this->footprint = prevVertex.ext().footprint.add_segment(
			float(pdf.forw), prevVertex.is_orthographic(), 0.0f, outCosAbs, incident.distance, inCosAbs);
	}

	CUDA_FUNCTION void update(const IvcmPathVertex& thisVertex,
							  const scene::Direction& excident,
							  const math::PdfPair pdf) {
		pdfBack = pdf.back;
	}
};

// incidentF/incidentB: per vertex area pdfs
// n: path length in segments
// idx: index of merge vertex
float get_mis_weight_photon(const AreaPdf* incidentF, const AreaPdf* incidentB, int n, int idx,
	float mergeArea, const float* reuseCount) {
	if(idx == 0 || idx == n) return 0.0f;
	// Start with camera connection
	float relPdfSumV = 1.0f / (float(incidentF[1]) * mergeArea * reuseCount[1]);
	// Collect merges and connections along view path
	for(int i = 1; i < idx; ++i) {
		float prevMerge = (incidentB[i] / incidentF[i+1]) * (reuseCount[i] / reuseCount[i+1]);
		float prevConnect = 1.0f / (float(incidentF[i+1]) * mergeArea * reuseCount[i+1]);
		relPdfSumV = prevConnect + prevMerge * (1.0f + relPdfSumV);
	}
	// Collect merges/connect/hit along light path
	float relPdfSumL = 1.0f / (reuseCount[n-1] * mergeArea * float(incidentB[n-1]));	// Connection
	relPdfSumL += incidentF[n] / incidentB[n] * relPdfSumL;						// Random hit
	for(int i = n-2; i >= idx; --i) {
		float prevMerge = (incidentF[i+1] / incidentB[i]) * (reuseCount[i+1] / reuseCount[i]);
		float prevConnect = 1.0f / (float(incidentB[i]) * mergeArea * reuseCount[i]);
		relPdfSumL = prevConnect + prevMerge * (1.0f + relPdfSumL);
	}
	return 1.0f / (1.0f + relPdfSumV + relPdfSumL);
}

// incidentF/incidentB: per vertex area pdfs
// n: path length in segments
// idx: index of first connection vertex
float get_mis_weight_connect(const AreaPdf* incidentF, const AreaPdf* incidentB, int n, int idx,
	float mergeArea, const float* reuseCount) {
	float relPdfSumV = 0.0f;
	// Collect merges and connections along view path
	for(int i = 1; i <= idx; ++i) {
		float prevConnect = incidentB[i] / incidentF[i];
		float curMerge = float(incidentB[i]) * mergeArea * reuseCount[i];
		relPdfSumV = curMerge + prevConnect * (1.0f + relPdfSumV);
	}
	// Collect merges/connect/hit along light path
	float relPdfSumL = incidentF[n] / incidentB[n];		// Random hit
	for( int i = n-1; i > idx; --i) {
		float prevConnect = incidentF[i] / incidentB[i];
		float curMerge = float(incidentF[i]) * mergeArea * reuseCount[i];
		relPdfSumL = curMerge + prevConnect * (1.0f + relPdfSumL);
	}
	return 1.0f / (1.0f + relPdfSumV + relPdfSumL);
}

float get_mis_weight_rhit(const AreaPdf* incidentF, const AreaPdf* incidentB, int n,
	float mergeArea, const float* reuseCount) {
	// Collect all connects/merges along the view path only
	float relPdfSumV = 0.0f;
	for(int i = 1; i < n; ++i) {
		float prevConnect = incidentB[i] / incidentF[i];
		float curMerge = float(incidentB[i]) * mergeArea * reuseCount[i];
		relPdfSumV = curMerge + prevConnect * (1.0f + relPdfSumV);
	}
	float connectionRel = incidentB[n] / incidentF[n];
	relPdfSumV = connectionRel * (1.0f + relPdfSumV);
	return 1.0f / (1.0f + relPdfSumV);
}

// Fill a range in the incidentF/B arrays
void copy_path_values(AreaPdf* incidentF, AreaPdf* incidentB,
	const IvcmPathVertex* vert, Interaction prevType,
	AngularPdf pdfBack, ConnectionDir connectionDir,
	int begin, int end) {
	int step = begin < end ? 1 : -1;
	for(int i = begin; i != end; i += step) {
		incidentF[i] = vert->ext().incidentPdf;
		incidentB[i] = vert->convert_pdf(prevType, pdfBack, connectionDir).pdf;
		pdfBack = vert->ext().pdfBack;
		connectionDir = vert->get_incident_connection();
		prevType = vert->get_type();
		vert = vert->previous();
	}
}


struct ConnectionValue { Spectrum bxdfs; float cosines; };
ConnectionValue connect(const IvcmPathVertex& path0, const IvcmPathVertex& path1,
						const scene::SceneDescriptor<Device::CPU>& scene,
						Pixel& coord, float mergeArea, const float* reuseCount,
						AreaPdf* incidentF, AreaPdf* incidentB
) {
	// Some vertices will always have a contribution of 0 if connected (e.g. directional light with camera).
	if(!IvcmPathVertex::is_connection_possible(path0, path1)) return {Spectrum{0.0f}, 0.0f};
	Connection connection = IvcmPathVertex::get_connection(path0, path1);
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
				scene,connection.v0, path1.get_position(connection.v0),
				path0.get_geometric_normal(),  path1.get_geometric_normal(), connection.dir)) {
			bxdfProd *= scene.media[path0.get_medium(connection.dir)].get_transmission(connection.distance);
			// Collect quantities for MIS
			int pl0 = path0.get_path_len();
			incidentB[pl0] = path0.convert_pdf(path1.get_type(), val1.pdf.forw, connection).pdf;
			incidentF[pl0] = path0.ext().incidentPdf;
			copy_path_values(incidentF, incidentB, path0.previous(), path0.get_type(),
				val0.pdf.back, path0.get_incident_connection(), pl0-1, -1);
			int pathLen = pl0 + path1.get_path_len() + 1;
			incidentF[pl0+1] = path1.convert_pdf(path0.get_type(), val0.pdf.forw, connection).pdf;
			incidentB[pl0+1] = path1.ext().incidentPdf;
			copy_path_values(incidentB, incidentF, path1.previous(), path1.get_type(),
				val1.pdf.back, path1.get_incident_connection(), pl0 + 2, pathLen + 1);
			float misWeight = get_mis_weight_connect(incidentF, incidentB, pathLen, pl0, mergeArea, reuseCount);
			return {bxdfProd * (misWeight / connection.distanceSq), cosProd};
		}
	}
	return {Spectrum{0.0f}, 0.0f};
}

Spectrum merge(const IvcmPathVertex& viewPath, const IvcmPathVertex& photon,
			   const scene::SceneDescriptor<Device::CPU>& scene,
			   float mergeArea, const float* reuseCount,
			   AreaPdf* incidentF, AreaPdf* incidentB) {
	// Radiance estimate
	Pixel tmpCoord;
	scene::Direction geoNormal = photon.get_geometric_normal();
	auto bsdf = viewPath.evaluate(-photon.get_incident_direction(),
								   scene.media, tmpCoord, false,
								   &geoNormal);
	// Collect values for mis
	int pl0 = viewPath.get_path_len();
	int pathLen = pl0 + photon.get_path_len();
	incidentF[pl0] = viewPath.ext().incidentPdf;
	incidentB[pl0] = photon.ext().incidentPdf;
	copy_path_values(incidentF, incidentB, viewPath.previous(), viewPath.get_type(),
		bsdf.pdf.back, viewPath.get_incident_connection(), pl0 - 1, -1);
	copy_path_values(incidentB, incidentF, photon.previous(), photon.get_type(),
		bsdf.pdf.forw, photon.get_incident_connection(), pl0 + 1, pathLen + 1);
	float misWeight = get_mis_weight_photon(incidentF, incidentB, pathLen, pl0, mergeArea, reuseCount);
	return bsdf.value * photon.ext().throughput * misWeight;
}

} // namespace ::


CpuIvcm::CpuIvcm() {}
CpuIvcm::~CpuIvcm() {}

void CpuIvcm::post_reset() {
	ResetEvent resetFlags { get_reset_event().is_set(ResetEvent::RENDERER_ENABLE) ?
								ResetEvent::ALL : get_reset_event() };
	init_rngs(m_outputBuffer.get_num_pixels());
	if(resetFlags.resolution_changed()) {
		m_photonMapManager.resize(m_outputBuffer.get_num_pixels() * m_params.maxPathLength);
		m_photonMap = m_photonMapManager.acquire<Device::CPU>();
		m_pathEndPoints.resize(m_outputBuffer.get_num_pixels());
	}
	if(resetFlags.is_set(ResetEvent::PARAMETER)) {
		m_tmpPathProbabilities.resize(get_thread_num() * 2 * (m_params.maxPathLength + 1));
		m_tmpViewPathVertices.resize(get_thread_num() * (m_params.maxPathLength + 1));
		m_tmpReuseCounts.resize(get_thread_num() * (m_params.maxPathLength + 1));
	}
	// TODO: reasonable density structure capacities
	if(resetFlags.geometry_changed())
		m_density = std::make_unique<data_structs::DmOctree>(m_sceneDesc.aabb,
			1024 * 1024 * 4, 8.0f, true);//*/
	/*if(resetFlags.is_set(ResetEvent::RENDERER_ENABLE))
		m_density = std::make_unique<data_structs::DmHashGrid>(1024 * 1024);
	m_density->set_cell_size(m_params.mergeRadius * m_sceneDesc.diagSize * 2.0001f);//*/
	m_density->clear();

	m_density2 = std::make_unique<data_structs::KdTree<char,3>>();
	m_density2->reserve(1024 * 1024);
}


float CpuIvcm::get_density(const ei::Vec3& pos, const ei::Vec3& normal, float currentMergeRadius) const {
	//return m_density->get_density(pos, normal);
	return m_density->get_density_interpolated(pos, normal);
	/*currentMergeRadius *= currentMergeRadius;
	auto it = m_photonMap.find_first(pos);
	int count = 0;
	while(it) {
		if(lensq(it->get_position() - pos) < currentMergeRadius)
			++count;
		++it;
	}
	return count / (ei::PI*currentMergeRadius);//*/
	/*int idx[5];
	float rSq[5];
	m_density2->query_euclidean(pos, 5, idx, rSq, currentMergeRadius * 10.0f);
	int count = 0;
	float kernelSum = 0.0f;
	while(idx[count] != -1 && count < 4) {
		kernelSum += 3.0f * ei::sq(1.0f - rSq[count] / rSq[4]);
		++count;
	}
	return kernelSum / (ei::PI*rSq[4]);
	if(idx[count] != -1) ++count;
	return ei::max(0,count-1) / (ei::PI*rSq[4]);//*/
}

void CpuIvcm::iterate() {
	auto scope = Profiler::instance().start<CpuProfileState>("CPU IVCM iteration", ProfileLevel::LOW);

	float currentMergeRadius = m_params.mergeRadius * m_sceneDesc.diagSize;
	if(m_params.progressive)
		currentMergeRadius *= powf(float(m_currentIteration + 1), -1.0f / 6.0f);
	m_photonMap.clear(currentMergeRadius * 2.0001f);
	m_density->set_iteration(m_currentIteration + 1);
	//m_density2->clear();

	// First pass: Create one photon path per view path
	u64 photonSeed = m_rngs[0].next();
	int numPhotons = m_outputBuffer.get_num_pixels();
/*#pragma PARALLEL_FOR
	for(int i = 0; i < numPhotons; ++i) {
		this->trace_photon(i, numPhotons, photonSeed, currentMergeRadius);
	}
	//m_density->balance();

	if(needs_density()) {
		const int n = m_photonMap.size();
		//m_density2->build();
#pragma PARALLEL_FOR
		for(int i = 0; i < n; ++i) {
			IvcmPathVertex& photon = m_photonMap.get_data_by_index(i);
			photon.ext().density = get_density(photon.get_position(), photon.get_geometric_normal(), currentMergeRadius);
		}
	}*/

	// Second pass: trace view paths and merge
#pragma PARALLEL_FOR
	for(int pixel = 0; pixel < m_outputBuffer.get_num_pixels(); ++pixel) {
		AreaPdf* incidentF = m_tmpPathProbabilities.data() + get_current_thread_idx() * 2 * (m_params.maxPathLength + 1);
		AreaPdf* incidentB = incidentF + (m_params.maxPathLength + 1);
		IvcmPathVertex* vertexBuffer = m_tmpViewPathVertices.data() + get_current_thread_idx() * (m_params.maxPathLength + 1);
		float* reuseCount = m_tmpReuseCounts.data() + get_current_thread_idx() * (m_params.maxPathLength + 1);
		this->sample(Pixel{ pixel % m_outputBuffer.get_width(), pixel / m_outputBuffer.get_width() },
					 pixel, numPhotons, currentMergeRadius, incidentF, incidentB, vertexBuffer, reuseCount);
	}

	logInfo("[CpuIvcm::iterate] Density structure memory: ", m_density->mem_size() / (1024 * 1024), "MB, ",
		ei::round((1000.0f * m_density->size()) / m_density->capacity()) / 10.0f, "%");
}


void CpuIvcm::trace_photon(int idx, int numPhotons, u64 seed, float currentMergeRadius) {
	math::RndSet2 rndStart { m_rngs[idx].next() };
	//u64 lightTreeRnd = m_rngs[idx].next();
	scene::lights::Emitter p = scene::lights::emit(m_sceneDesc, idx, numPhotons, seed, rndStart);
	IvcmPathVertex vertex;
	IvcmPathVertex::create_light(&vertex, nullptr, p);
	const IvcmPathVertex* previous = m_photonMap.insert(p.initVec, vertex);
	math::Throughput throughput;
	float mergeArea = ei::PI * currentMergeRadius * currentMergeRadius;

	int pathLen = 0;
	while(pathLen < m_params.maxPathLength-1) { // -1 because there is at least one segment on the view path
		// Walk
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		VertexSample sample;
		if(walk(m_sceneDesc, *previous, rnd, rndRoulette, true, throughput, vertex, sample) != WalkResult::HIT)
			break;
		++pathLen;

		// Store a photon to the photon map
		previous = m_photonMap.insert(vertex.get_position(), vertex);
		m_density->increase_count(vertex.get_position(), vertex.get_geometric_normal());
		//m_density->increase_count(vertex.get_position());
		//m_density2->insert(vertex.get_position(), 0);
	}

	m_pathEndPoints[idx] = previous;
}

void CpuIvcm::sample(const Pixel coord, int idx, int numPhotons, float currentMergeRadius,
					 AreaPdf* incidentF, AreaPdf* incidentB, IvcmPathVertex* vertexBuffer,
					 float* reuseCount) {
	float mergeRadiusSq = currentMergeRadius * currentMergeRadius;
	float mergeArea = ei::PI * mergeRadiusSq;
	u64 lightPathIdx = cn::WangHash{}(idx) % numPhotons;
	// Trace view path
	// Create a start for the path
	IvcmPathVertex* currentVertex = vertexBuffer;
	IvcmPathVertex::create_camera(currentVertex, nullptr, m_sceneDesc.camera.get(), coord, m_rngs[idx].next());
	math::Throughput throughput;
	int viewPathLen = 0;
	do {
		// Make a connection to any event on the light path
	/*	const IvcmPathVertex* lightVertex = m_pathEndPoints[lightPathIdx];
		if(!m_params.showDensity) while(lightVertex) {
			int lightPathLen = lightVertex->get_path_len();
			int pathLen = lightPathLen + 1 + viewPathLen;
			if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength) {
				Pixel outCoord = coord;
				compute_counts(reuseCount, mergeArea, numPhotons, false, currentVertex, viewPathLen, lightVertex, lightPathLen);
				auto conVal = connect(*currentVertex, *lightVertex, m_sceneDesc, outCoord, mergeArea, reuseCount, incidentF, incidentB);
				mAssert(!isnan(conVal.cosines) && !isnan(conVal.bxdfs.x) && !isnan(throughput.weight.x) && !isnan(currentVertex->ext().throughput.x));
				m_outputBuffer.contribute(outCoord, throughput, math::Throughput{lightVertex->ext().throughput, 1.0f}, conVal.cosines, conVal.bxdfs);
			}
			lightVertex = lightVertex->previous();
		}//*/

		// Walk
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		VertexSample sample;
		const WalkResult walkRes = walk(m_sceneDesc, *currentVertex, rnd, rndRoulette, false, throughput,
										*(currentVertex + 1), sample);
		if(walkRes == WalkResult::CANCEL)
			break;
		++viewPathLen;
		++currentVertex;

		if(walkRes == WalkResult::HIT) {
			float c = scene::accel_struct::compute_face_curvature(m_sceneDesc, currentVertex->get_primitive_id(), currentVertex->get_geometric_normal());
			m_outputBuffer.contribute(coord, throughput, Spectrum{c}, scene::Point{0.0f},
				scene::Direction{0.0f}, Spectrum{0.0f});
		}
		return;

		if(needs_density())
			currentVertex->ext().density = get_density(currentVertex->get_position(), currentVertex->get_geometric_normal(), currentMergeRadius);

		// Visualize density map (disables all other contributions)
		if(m_params.showDensity && walkRes == WalkResult::HIT) {
			m_outputBuffer.set(coord, 0, Spectrum{currentVertex->ext().density * (m_currentIteration + 1)});
			//m_outputBuffer.contribute(coord, throughput, Spectrum{currentVertex->ext().density}, currentVertex->get_position(),
			//							currentVertex->get_normal(), currentVertex->get_albedo());
			break;
		}//*/
		// Evaluate direct hit of area ligths and the background
		if(viewPathLen >= m_params.minPathLength) {
			EmissionValue emission = currentVertex->get_emission(m_sceneDesc, currentVertex->previous()->get_position());
			if(emission.value != 0.0f) {
				incidentF[viewPathLen] = currentVertex->ext().incidentPdf;
				incidentB[viewPathLen] = emission.emitPdf;
				copy_path_values(incidentF, incidentB, currentVertex->previous(), currentVertex->get_type(),
					emission.pdf, currentVertex->get_incident_connection(), viewPathLen - 1, -1);
				compute_counts(reuseCount, mergeArea, numPhotons, false, currentVertex, viewPathLen-1, nullptr, 0);
				float misWeight = get_mis_weight_rhit(incidentF, incidentB, viewPathLen, mergeArea, reuseCount);
				emission.value *= misWeight;
			}
			mAssert(!isnan(emission.value.x));
			m_outputBuffer.contribute(coord, throughput, emission.value, currentVertex->get_position(),
									currentVertex->get_normal(), currentVertex->get_albedo());
		}//*/
		if(currentVertex->is_end_point()) break;

		// Merges
		Spectrum radiance { 0.0f };
		scene::Point currentPos = currentVertex->get_position();
		auto photonIt = m_photonMap.find_first(currentPos);
		while(photonIt) {
			auto& photon = *photonIt;
			// Only merge photons which are within the sphere around our position.
			// and which have the correct full path length.
			int lightPathLen = photon.get_path_len();
			int pathLen = viewPathLen + lightPathLen;
			if(lightPathLen > 0 && pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength
				&& lensq(photon.get_position() - currentPos) < mergeRadiusSq) {
				compute_counts(reuseCount, mergeArea, numPhotons, true, currentVertex, viewPathLen, &photon, lightPathLen);
				//reuseCount[viewPathLen] = numPhotons / ei::max(1.0f, 0.5f * (currentVertex->ext().density + photon.ext().density) * mergeArea);
				radiance += merge(*currentVertex, photon, m_sceneDesc, mergeArea, reuseCount,
								  incidentF, incidentB);
			}
			++photonIt;
		}
		radiance /= mergeArea * numPhotons;
		m_outputBuffer.contribute(coord, throughput, radiance, scene::Point{0.0f},
			scene::Direction{0.0f}, Spectrum{0.0f});//*/
	} while(viewPathLen < m_params.maxPathLength);
}

void CpuIvcm::compute_counts(float* reuseCount, float mergeArea, int numPhotons, bool merge,
							 const IvcmPathVertex* path0, int pl0,
							 const IvcmPathVertex* path1, int pl1) {
	int pl = pl0 + pl1 + (merge ? 0 : 1);
	reuseCount[0] = reuseCount[pl] = 0.0f;
	switch(m_params.heuristic) {
		case PHeuristic::Values::VCM: {
			for(int i = 1; i < pl; ++i)
				reuseCount[i] = static_cast<float>(numPhotons);
		} break;
		case PHeuristic::Values::VCMPlus: {
			if(merge) {
				float expected = mergeArea * 0.5f * (path0->ext().density + path1->ext().density);
				reuseCount[pl0] = numPhotons / ei::max(1.0f, expected);
				path0 = path0->previous();
				path1 = path1->previous();
			}
			for(int i = merge ? pl0-1 : pl0; i > 0; --i) {
				float expected = mergeArea * path0->ext().density;
				reuseCount[i] = numPhotons / ei::max(1.0f, expected);
				path0 = path0->previous();
			}
			for(int i = pl0+1; i < pl; ++i) {
				float expected = mergeArea * path1->ext().density;
				reuseCount[i] = numPhotons / ei::max(1.0f, expected);
				path1 = path1->previous();
			}
		} break;
		case PHeuristic::Values::VCMStar: {
			float divisorSum = 0.0f;
			if(merge) {
				float expected = mergeArea * 0.5f * (path0->ext().density + path1->ext().density);
				float divisor = 1.0f / ei::max(1.0f, expected);
				reuseCount[pl0] = numPhotons * divisor;
				divisorSum += divisor;
				path0 = path0->previous();
				path1 = path1->previous();
			}
			for(int i = merge ? pl0-1 : pl0; i > 0; --i) {
				float expected = mergeArea * path0->ext().density;
				float divisor = 1.0f / ei::max(1.0f, expected);
				reuseCount[i] = numPhotons * divisor;
				divisorSum += divisor;
				path0 = path0->previous();
			}
			for(int i = pl0+1; i < pl; ++i) {
				float expected = mergeArea * path1->ext().density;
				float divisor = 1.0f / ei::max(1.0f, expected);
				reuseCount[i] = numPhotons * divisor;
				divisorSum += divisor;
				path1 = path1->previous();
			}
			float norm = (pl - 1) / divisorSum;
			for(int i = 1; i < pl; ++i)
				reuseCount[i] *= norm;
		} break;
	}
}

void CpuIvcm::init_rngs(int num) {
	m_rngs.resize(num);
	int seed = m_params.seed * (num + 1);
	for(int i = 0; i < num; ++i)
		m_rngs[i] = math::Rng(i + seed);
}

} // namespace mufflon::renderer
