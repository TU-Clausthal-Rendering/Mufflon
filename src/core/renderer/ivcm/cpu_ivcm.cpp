#include "cpu_ivcm.hpp"
#include "profiler/cpu_profiler.hpp"
#include "util/parallel.hpp"
#include "core/cameras/camera.hpp"
#include "core/math/rng.hpp"
#include "core/renderer/path_util.hpp"
#include "core/renderer/random_walk.hpp"
#include "core/scene/materials/medium.hpp"
#include "core/scene/accel_structs/intersection.hpp"
#include "core/scene/lights/light_tree_sampling.hpp"

#include <cn/rnd.hpp>
#include <cmath>

namespace mufflon::renderer {

namespace {

static float s_curvScale;
static int s_numPhotons;

} // namespace

CUDA_FUNCTION void IvcmVertexExt::init(const IvcmPathVertex& /*thisVertex*/,
									   const AreaPdf inAreaPdf,
									   const AngularPdf inDirPdf,
									   const float pChoice) {
	this->incidentPdf = VertexExtension::mis_start_pdf(inAreaPdf, inDirPdf, pChoice);
	this->throughput = Spectrum{ 1.0f };
	float sourceCount = 1.0f;//pChoice * 800 * 600;
	this->footprint.init(1.0f / (float(inAreaPdf) * sourceCount), 1.0f / (float(inDirPdf) * sourceCount), pChoice);
}


CUDA_FUNCTION void IvcmVertexExt::update(const IvcmPathVertex& prevVertex,
										 const IvcmPathVertex& thisVertex,
										 const math::PdfPair pdf,
										 const Connection& incident,
										 const Spectrum& throughput,
										 const float /*continuationPropability*/,
										 const Spectrum& /*transmission*/,
										 const scene::SceneDescriptor<Device::CPU>& scene,
										 int numPhotons) {
	float inCos = thisVertex.get_geometric_factor(incident.dir);
	float outCos = prevVertex.get_geometric_factor(incident.dir);
	bool orthoConnection = prevVertex.is_orthographic() || thisVertex.is_orthographic();
	this->incidentPdf = VertexExtension::mis_pdf(pdf.forw, orthoConnection, incident.distance, ei::abs(inCos));
	this->throughput = throughput;
	if(prevVertex.is_hitable()) {
		// Compute as much as possible from the conversion factor.
		// At this point we do not know n and A for the photons. This quantities
		// are added in the kernel after the walk.
		this->prevConversionFactor = ei::abs(orthoConnection ? outCos : outCos / incident.distanceSq);
	}
	this->curvature = 0.0f; // Mean curvature for the footprint
	if(prevVertex.get_primitive_id().is_valid()) {
		this->curvature = scene::accel_struct::fetch_curvature(scene,
															   prevVertex.get_primitive_id(),
															   prevVertex.get_surface_params(),
															   prevVertex.get_geometric_normal())
			* s_curvScale;
	}
	float pdfForw = float(pdf.forw);
	if(prevVertex.is_camera())
		pdfForw *= numPhotons; // == numPixels
	auto prevEta = prevVertex.get_eta(scene.media);
	this->footprint = prevVertex.ext().footprint.add_segment(
		pdfForw, prevVertex.is_orthographic(), this->curvature, prevEta.inCos, outCos,
		prevEta.eta, incident.distance, inCos, 1.0f);
}

CUDA_FUNCTION void IvcmVertexExt::update(const IvcmPathVertex& /*thisVertex*/,
										 const scene::Direction& /*excident*/,
										 const VertexSample& sample,
										 const scene::SceneDescriptor<Device::CPU>& /*scene*/,
										 int /*numPhotons*/) {
	pdfBack = sample.pdf.back;
}

// A wrapper to provide the vertex's current pdf without changing the original vertex.
class VertexWrapper {
	const IvcmPathVertex* m_vertex;
	math::PdfPair m_pdfs;
public:
	VertexWrapper(const IvcmPathVertex* vertex, const math::PdfPair& pdfs) :
		m_vertex(vertex),
		m_pdfs(pdfs)
	{}
	VertexWrapper(const IvcmPathVertex& vertex) :
		m_vertex(&vertex),
		m_pdfs{AngularPdf{0.0f}, vertex.ext().pdfBack}
	{}
	const AngularPdf& pdf_forw() const noexcept { return m_pdfs.forw; }
	const AngularPdf& pdf_back() const noexcept { return m_pdfs.back; }
	VertexWrapper previous() const {
		if(m_vertex->previous())
			return {*m_vertex->previous()};
		else return {nullptr, math::PdfPair{}};
	}
	float indist() const {
		return m_vertex ? m_vertex->get_incident_dist() : 0.0f;
	}
	scene::Direction indir() const {
		return m_vertex ? m_vertex->get_incident_direction() : scene::Direction{0.0f};
	}
	float cosine(const scene::Direction& dir) {
		return m_vertex ? m_vertex->get_geometric_factor(dir) : 0.0f;
	}
	bool is_orthographic() const noexcept { return m_vertex ? m_vertex->is_orthographic() : false; }
	const Footprint2D& footprint() const { return m_vertex->ext().footprint; }
	const IvcmVertexExt& ext() const { return m_vertex->ext(); }
	IvcmPathVertex::RefractionInfo eta(const scene::materials::Medium* media) const {
		return m_vertex ? m_vertex->get_eta(media) : IvcmPathVertex::RefractionInfo{1.0f, 0.0f};
	}
};

namespace {

// incidentF/incidentB: per vertex area pdfs
// n: path length in segments
// idx: index of merge vertex
float get_mis_weight_photon(const AreaPdf* incidentF, const AreaPdf* incidentB, int n, int idx,
	float mergeArea, const float* reuseCount) {
	if(idx == 0 || idx == n) return 0.0f;
	// Start with camera connection
	float relPdfSumV = 1.0f / (float(incidentF[1]) * mergeArea * reuseCount[1]); // = (1/(p * A * reuseCount[1]) * (reuseCount[1]/s_numPhotons)
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
	//	if(i == 1) prevConnect *= reuseCount[1] / s_numPhotons;		// LT reuse
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
	//if(idx == 0)
	//	relPdfSumL *= s_numPhotons / reuseCount[1];	// LT reuse 2
	return 1.0f / (1.0f + relPdfSumV + relPdfSumL);
}

float get_mis_weight_rhit(const AreaPdf* incidentF, const AreaPdf* incidentB, int n,
	float mergeArea, const float* reuseCount) {
	// Collect all connects/merges along the view path only
	float relPdfSumV = 0.0f;
	for(int i = 1; i < n; ++i) {
		float prevConnect = incidentB[i] / incidentF[i];
	//	if(i == 1) prevConnect *= reuseCount[1] / s_numPhotons;		// LT reuse
	//	float curMerge = (i == n) ? 0.0f : float(incidentB[i]) * mergeArea * reuseCount[i];
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

} // namespace ::


CpuIvcm::ConnectionValue
CpuIvcm::connect(const IvcmPathVertex& path0, const IvcmPathVertex& path1,
						Pixel& coord, float mergeArea, int numPhotons, float* reuseCount,
						AreaPdf* incidentF, AreaPdf* incidentB
) {
	// Some vertices will always have a contribution of 0 if connected (e.g. directional light with camera).
	if(!IvcmPathVertex::is_connection_possible(path0, path1)) return {Spectrum{0.0f}, 0.0f};
	Connection connection = IvcmPathVertex::get_connection(path0, path1);
	auto val0 = path0.evaluate( connection.dir, m_sceneDesc.media, coord, false);
	auto val1 = path1.evaluate(-connection.dir, m_sceneDesc.media, coord, true);
	// Cancel reprojections outside the screen
	if(coord.x == -1) return {Spectrum{0.0f}, 0.0f};
	Spectrum bxdfProd = val0.value * val1.value;
	float cosProd = val0.cosOut * val1.cosOut;//TODO: abs?
	mAssert(cosProd >= 0.0f);
	mAssert(!std::isnan(bxdfProd.x));
	// Early out if there would not be a contribution (estimating the materials is usually
	// cheaper than the any-hit test).
	if(any(greater(bxdfProd, 0.0f)) && cosProd > 0.0f) {
		// Shadow test
		if(!scene::accel_struct::any_intersection(
				m_sceneDesc,connection.v0, path1.get_position(connection.v0),
				path0.get_geometric_normal(),  path1.get_geometric_normal(), connection.dir)) {
			bxdfProd *= m_sceneDesc.media[path0.get_medium(connection.dir)].get_transmission(connection.distance);
			// Compute reuse counts with varaying algorithms
			int pl0 = path0.get_path_len();
			int pl1 = path1.get_path_len();
			math::PdfPair v0pdfs = val0.pdf;
			if(path0.is_camera())
				v0pdfs.forw *= float(numPhotons); // == numPixels!
			compute_counts(reuseCount, mergeArea, numPhotons, connection.distance, connection.dir,
				VertexWrapper{&path0, v0pdfs}, pl0, VertexWrapper{&path1,val1.pdf}, pl1);

			// Collect quantities for MIS
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


Spectrum CpuIvcm::merge(const IvcmPathVertex& viewPath, const IvcmPathVertex& photon,
			   float mergeArea, int numPhotons, float* reuseCount,
			   AreaPdf* incidentF, AreaPdf* incidentB) {
	// Radiance estimate
	Pixel tmpCoord;
	scene::Direction geoNormal = photon.get_geometric_normal();
	auto bsdf = viewPath.evaluate(-photon.get_incident_direction(),
								   m_sceneDesc.media, tmpCoord, false,
								   &geoNormal);
	// Early out if no contribution
	if(bsdf.value == Spectrum{0.0f})
		return Spectrum{0.0f};

	// Compute reuse factors with varaying methods.
	int pl0 = viewPath.get_path_len();
	int pl1 = photon.get_path_len();
	compute_counts(reuseCount, mergeArea, numPhotons, 0.0f, scene::Direction{0.0f},
		VertexWrapper{&viewPath,bsdf.pdf}, pl0,
		VertexWrapper{&photon,{bsdf.pdf.back, bsdf.pdf.forw}}, pl1);
	// Collect values for mis
	int pathLen = pl0 + pl1;
	incidentF[pl0] = viewPath.ext().incidentPdf;
	incidentB[pl0] = photon.ext().incidentPdf;
	copy_path_values(incidentF, incidentB, viewPath.previous(), viewPath.get_type(),
		bsdf.pdf.back, viewPath.get_incident_connection(), pl0 - 1, -1);
	copy_path_values(incidentB, incidentF, photon.previous(), photon.get_type(),
		bsdf.pdf.forw, photon.get_incident_connection(), pl0 + 1, pathLen + 1);
	float misWeight = get_mis_weight_photon(incidentF, incidentB, pathLen, pl0, mergeArea, reuseCount);
	return bsdf.value * photon.ext().throughput * misWeight;
}




CpuIvcm::CpuIvcm(mufflon::scene::WorldContainer& world) :
	RendererBase<Device::CPU, IvcmTargets>(world, std::nullopt, {
		scene::AttributeIdentifier{scene::AttributeType::FLOAT, "mean_curvature"}
	}, {}, {})
{}
CpuIvcm::~CpuIvcm() {}

void CpuIvcm::pre_reset() {
	m_currentScene->compute_curvature();
}

void CpuIvcm::post_reset() {
	ResetEvent resetFlags { { get_reset_event().is_set(ResetEvent::RENDERER_ENABLE) ?
								ResetEvent::ALL : get_reset_event() } };
	init_rngs(m_outputBuffer.get_num_pixels());
	if(resetFlags.resolution_changed()) {
		m_photonMapManager.resize(m_outputBuffer.get_num_pixels() * m_params.maxPathLength);
		m_photonMap = m_photonMapManager.acquire<Device::CPU>();
		m_pathEndPoints.resize(m_outputBuffer.get_num_pixels());
	}
	if(resetFlags.is_set(ResetEvent::PARAMETER)) {
		if(!resetFlags.resolution_changed()) {
			m_photonMapManager.resize(m_outputBuffer.get_num_pixels() * m_params.maxPathLength);
			m_photonMap = m_photonMapManager.acquire<Device::CPU>();
		}
		m_tmpPathProbabilities.resize(get_thread_num() * 2 * (m_params.maxPathLength + 1));
		m_tmpViewPathVertices.resize(get_thread_num() * (m_params.maxPathLength + 1));
		m_tmpReuseCounts.resize(get_thread_num() * (m_params.maxPathLength + 1));
	}
	// TODO: reasonable density structure capacities
	if(resetFlags.geometry_changed())
		m_density = std::make_unique<data_structs::DmOctree<>>(m_sceneDesc.aabb,
			1024 * 1024 * 4, 8.0f, true);//*/
	/*if(resetFlags.is_set(ResetEvent::RENDERER_ENABLE))
		m_density = std::make_unique<data_structs::DmHashGrid>(1024 * 1024);
	m_density->set_cell_size(m_params.mergeRadius * m_sceneDesc.diagSize * 2.0001f);//*/
	m_density->clear();

	//m_density2 = std::make_unique<data_structs::KdTree<char,3>>();
	//m_density2->reserve(1024 * 1024);
	s_curvScale = m_params.m_curvScale;
}


float CpuIvcm::get_density(const ei::Vec3& pos, const ei::Vec3& normal, float /*currentMergeRadius*/) const {
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
	auto scope = Profiler::core().start<CpuProfileState>("CPU IVCM iteration", ProfileLevel::LOW);

	float currentMergeRadius = m_params.mergeRadius * m_sceneDesc.diagSize;
	if(m_params.progressive)
		currentMergeRadius *= powf(float(m_currentIteration + 1), -1.0f / 6.0f);
	m_photonMap.clear(currentMergeRadius * 2.0001f);
	m_density->set_iteration(m_currentIteration + 1);
	//m_density2->clear();

	// First pass: Create one photon path per view path
	u64 photonSeed = m_rngs[0].next();
	int numPhotons = m_outputBuffer.get_num_pixels();
	s_numPhotons = numPhotons;
#pragma PARALLEL_FOR
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
	}

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

	if(needs_density() || m_outputBuffer.is_target_enabled<DensityTarget>()) {
		logInfo("[CpuIvcm::iterate] Density structure memory: ", m_density->mem_size() / (1024 * 1024), "MB, ",
			ei::round((1000.0f * m_density->size()) / m_density->capacity()) / 10.0f, "%");
	}
}


void CpuIvcm::trace_photon(int idx, int numPhotons, u64 /*seed*/, float /*currentMergeRadius*/) {
	math::RndSet2 rndStart { m_rngs[idx].next() };
	u64 lightTreeRnd = m_rngs[idx].next();
	scene::lights::Emitter p = scene::lights::emit(m_sceneDesc, idx, numPhotons, lightTreeRnd, rndStart);
	IvcmPathVertex vertex;
	IvcmPathVertex::create_light(&vertex, nullptr, p);
	const IvcmPathVertex* previous = m_photonMap.insert(p.initVec, vertex);
	Spectrum throughput { 1.0f };
	//float mergeArea = ei::PI * currentMergeRadius * currentMergeRadius;

	int pathLen = 0;
	while(pathLen < m_params.maxPathLength-1) { // -1 because there is at least one segment on the view path
		// Walk
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		VertexSample sample;
		if(walk(m_sceneDesc, *previous, rnd, rndRoulette, true, throughput, vertex, sample, m_sceneDesc, numPhotons) != WalkResult::HIT)
			break;
		++pathLen;

		// Store a photon to the photon map
		previous = m_photonMap.insert(vertex.get_position(), vertex);
		if(needs_density() || m_outputBuffer.is_target_enabled<DensityTarget>()) {
			m_density->increase_count(vertex.get_position(), vertex.get_geometric_normal());
			//m_density->increase_count(vertex.get_position());
			//m_density2->insert(vertex.get_position(), 0);
		}
	}

	m_pathEndPoints[idx] = previous;
}

void CpuIvcm::sample(const Pixel coord, int idx, int numPhotons, float currentMergeRadius,
					 AreaPdf* incidentF, AreaPdf* incidentB, IvcmPathVertex* vertexBuffer,
					 float* reuseCount) {
	float mergeRadiusSq = currentMergeRadius * currentMergeRadius;
	float mergeArea = ei::PI * mergeRadiusSq;
	//u64 lightPathIdx = cn::WangHash{}(idx) % numPhotons;
	//u64 lightPathIdx = (i64(idx) * 2147483647ll) % numPhotons;
	u64 lightPathIdx = idx;
	// Trace view path
	// Create a start for the path
	IvcmPathVertex* currentVertex = vertexBuffer;
	IvcmPathVertex::create_camera(currentVertex, nullptr, m_sceneDesc.camera.get(), coord, m_rngs[idx].next());
	Spectrum throughput { 1.0f };
	int viewPathLen = 0;
	do {
		// Make a connection to any event on the light path
		const IvcmPathVertex* lightVertex = m_pathEndPoints[lightPathIdx];
		while(lightVertex) {
			int lightPathLen = lightVertex->get_path_len();
			int pathLen = lightPathLen + 1 + viewPathLen;
			if(pathLen >= m_params.minPathLength && pathLen <= m_params.maxPathLength) {
				Pixel outCoord = coord;
				auto conVal = connect(*currentVertex, *lightVertex, outCoord, mergeArea, numPhotons, reuseCount, incidentF, incidentB);
				if(outCoord.x != -1) {
					mAssert(!std::isnan(conVal.cosines) && !std::isnan(conVal.bxdfs.x) && !std::isnan(throughput.x) && !std::isnan(currentVertex->ext().throughput.x));
					m_outputBuffer.contribute<RadianceTarget>(outCoord, throughput * lightVertex->ext().throughput * conVal.cosines * conVal.bxdfs);
				}
			}
			lightVertex = lightVertex->previous();
		}//*/

		// Walk
		math::RndSet2_1 rnd { m_rngs[idx].next(), m_rngs[idx].next() };
		float rndRoulette = math::sample_uniform(u32(m_rngs[idx].next()));
		VertexSample sample;
		const WalkResult walkRes = walk(m_sceneDesc, *currentVertex, rnd, rndRoulette, false, throughput,
										*(currentVertex + 1), sample, m_sceneDesc, numPhotons);
		if(walkRes == WalkResult::CANCEL)
			break;
		++viewPathLen;
		++currentVertex;

		/*if(walkRes == WalkResult::HIT) {
			float c = scene::accel_struct::fetch_curvature(m_sceneDesc,
				currentVertex->get_primitive_id(),
				currentVertex->get_surface_params(),
				currentVertex->get_geometric_normal());
			//float c = scene::accel_struct::compute_face_curvature(m_sceneDesc, currentVertex->get_primitive_id(), currentVertex->get_geometric_normal());
			c /= 2.0f;
			m_outputBuffer.contribute(coord, throughput, Spectrum{ei::max(0.0f, c), ei::abs(c) / 16.0f, ei::max(0.0f, -c)}, scene::Point{ei::abs(c)},
				scene::Direction{0.0f}, Spectrum{0.0f});
		} else m_outputBuffer.contribute(coord, throughput, Spectrum{1.0f}, scene::Point{0.0f},
					scene::Direction{0.0f}, Spectrum{0.0f});
		return;//*/

		if(needs_density() || m_outputBuffer.is_target_enabled<DensityTarget>())
			currentVertex->ext().density = get_density(currentVertex->get_position(), currentVertex->get_geometric_normal(), currentMergeRadius);

		// Visualize density map
		if(walkRes == WalkResult::HIT && viewPathLen == 1) {
			m_outputBuffer.contribute<DensityTarget>(coord, currentVertex->ext().density / numPhotons);
		}
		// Evaluate direct hit of area ligths and the background
		if(viewPathLen >= m_params.minPathLength) {
			EmissionValue emission = currentVertex->get_emission(m_sceneDesc, currentVertex->previous()->get_position());
			if(emission.value != 0.0f) {
				incidentF[viewPathLen] = currentVertex->ext().incidentPdf;
				incidentB[viewPathLen] = emission.emitPdf;
				auto inConnection = currentVertex->get_incident_connection();
				copy_path_values(incidentF, incidentB, currentVertex->previous(), currentVertex->get_type(),
					emission.pdf, inConnection, viewPathLen - 1, -1);
				IvcmPathVertex light;
				IvcmPathVertex::create_light(&light, nullptr, scene::lights::Emitter{
					currentVertex->get_position(),
					float(emission.emitPdf),
					emission.value, 1.0f,
					currentVertex->is_end_point() ? scene::lights::LightType::ENVMAP_LIGHT : scene::lights::LightType::AREA_LIGHT_QUAD,
					currentVertex->get_medium(inConnection.dir),
					{currentVertex->get_geometric_normal(), 0.0f}
				});
				compute_counts(reuseCount, mergeArea, numPhotons,
					currentVertex->get_incident_dist(), inConnection.dir,
					VertexWrapper{currentVertex->previous(), sample.pdf}, viewPathLen-1,
					VertexWrapper{&light, {emission.pdf, AngularPdf{0.0}}}, 0);
				float misWeight = get_mis_weight_rhit(incidentF, incidentB, viewPathLen, mergeArea, reuseCount);
				emission.value *= misWeight;
				m_outputBuffer.contribute<RadianceTarget>(coord, throughput * emission.value);
			}
			mAssert(!std::isnan(emission.value.x));
		}//*/
		if(currentVertex->is_end_point()) break;

		// Merges
		Spectrum radiance { 0.0f };
		float density = 0.0f;
		scene::Point currentPos = currentVertex->get_position();
		auto photonIt = m_photonMap.find_first(currentPos);
		float closestDensityMerge = mergeRadiusSq;
		while(photonIt) {
			auto& photon = *photonIt;
			// Only merge photons which are within the sphere around our position.
			// and which have the correct full path length.
			int lightPathLen = photon.get_path_len();
			int pathLen = viewPathLen + lightPathLen;
			if(lightPathLen > 0 && path_len_valid(pathLen)) {
				const float photonDist = lensq(photon.get_position() - currentPos);
				if(photonDist < mergeRadiusSq) {
					if(m_outputBuffer.is_target_enabled<RadianceTarget>()) {
						radiance += merge(*currentVertex, photon, mergeArea,
										  numPhotons, reuseCount, incidentF, incidentB);
					}
					if((viewPathLen == 1) && m_outputBuffer.is_target_enabled<FootprintTarget>()) {
						float area = photon.ext().footprint.get_area();
						// Correct incident normal
						scene::Direction photonDir = photon.get_incident_direction();
						float origCos = dot(photon.get_geometric_normal(), photonDir);
						float newCos = dot(currentVertex->get_geometric_normal(), photonDir);
						area *= ei::max(0.0f, origCos / newCos);
						if(area > 0.0f && dot(currentVertex->get_geometric_normal(), photon.get_geometric_normal()) > 0.0f) {
							mAssert(!std::isnan(area));
							if(photonDist < closestDensityMerge) {
							//if(1/area > density) {
								density = 1.0f / (area + 1e-8f);
								closestDensityMerge = photonDist;
							}
						}
					}
				}
			}
			++photonIt;
		}
		radiance /= mergeArea * numPhotons;

		m_outputBuffer.contribute<RadianceTarget>(coord, throughput * radiance);
		m_outputBuffer.contribute<FootprintTarget>(coord, density);
		//break;
	} while(viewPathLen < m_params.maxPathLength);
}


static float ivcm_heuristic(int numPhotons, float a0, float a1) {
	//return (a0 + a1) / (a0 + a1 / numPhotons);
	//float aR = (a0 * a0) / (a1 * a1);
	float aR = a0 / a1;
	if(aR > 1e12f) return 1.0f;
	//aR = pow(aR, 2.5f);
	aR *= aR;
	float count = (1.0f + aR) / (1.0f / numPhotons + aR);
	mAssert(count >= 1.0f);
//	mAssert(!std::isnan(count));
	return count;
}

void CpuIvcm::compute_counts(float* reuseCount, float mergeArea,
							 int numPhotons, float connectionDist,
							 const scene::Direction& connectionDir,
							 VertexWrapper path0, int pl0,
							 VertexWrapper path1, int pl1) {
	bool merge = connectionDist == 0.0f;
	int pl = pl0 + pl1 + (merge ? 0 : 1);
	reuseCount[0] = reuseCount[pl] = 0.0f;
	switch(m_params.heuristic) {
		case PHeuristic::Values::VCM: {
			for(int i = 1; i < pl; ++i)
				reuseCount[i] = static_cast<float>(numPhotons);
		} break;
		case PHeuristic::Values::VCMPlus: {
			if(merge) {
				float expected = mergeArea * 0.5f * (path0.ext().density + path1.ext().density);
				reuseCount[pl0] = numPhotons / ei::max(1.0f, expected);
				path0 = path0.previous();
				path1 = path1.previous();
			}
			for(int i = merge ? pl0-1 : pl0; i > 0; --i) {
				float expected = mergeArea * path0.ext().density;
				reuseCount[i] = numPhotons / ei::max(1.0f, expected);
				path0 = path0.previous();
			}
			for(int i = pl0+1; i < pl; ++i) {
				float expected = mergeArea * path1.ext().density;
				reuseCount[i] = numPhotons / ei::max(1.0f, expected);
				path1 = path1.previous();
			}
		} break;
		case PHeuristic::Values::VCMStar: {
			float divisorSum = 0.0f;
			if(merge) {
				float expected = mergeArea * 0.5f * (path0.ext().density + path1.ext().density);
				float divisor = 1.0f / ei::max(1.0f, expected);
				reuseCount[pl0] = numPhotons * divisor;
				divisorSum += divisor;
				path0 = path0.previous();
				path1 = path1.previous();
			}
			for(int i = merge ? pl0-1 : pl0; i > 0; --i) {
				float expected = mergeArea * path0.ext().density;
				float divisor = 1.0f / ei::max(1.0f, expected);
				reuseCount[i] = numPhotons * divisor;
				divisorSum += divisor;
				path0 = path0.previous();
			}
			for(int i = pl0+1; i < pl; ++i) {
				float expected = mergeArea * path1.ext().density;
				float divisor = 1.0f / ei::max(1.0f, expected);
				reuseCount[i] = numPhotons * divisor;
				divisorSum += divisor;
				path1 = path1.previous();
			}
			float norm = (pl - 1) / divisorSum;
			for(int i = 1; i < pl; ++i)
				reuseCount[i] *= norm;
		} break;
		case PHeuristic::Values::IVCM: {
			Footprint2D f0 = path0.footprint();
			Footprint2D f1 = path1.footprint();
			bool p0Ortho = path0.is_orthographic();
			bool p1Ortho = path1.is_orthographic();
			float p0Dist = connectionDist, p1Dist = connectionDist;
			float p0Pdf = float(path0.pdf_forw());
			float p1Pdf = float(path1.pdf_forw());
			float p0H = path0.ext().curvature;
			float p1H = path1.ext().curvature;
			auto p0Eta = path0.eta(m_sceneDesc.media);
			auto p1Eta = path1.eta(m_sceneDesc.media);
			scene::Direction p0OutDir = connectionDir;
			scene::Direction p1OutDir = -connectionDir;
			if(merge) {
				float a0 = path0.footprint().get_area();
				float a1 = path1.footprint().get_area();
				reuseCount[pl0] = ivcm_heuristic(numPhotons, a0, a1);
				p0Dist = path1.indist();
				p1Dist = path0.indist();
				p0OutDir = -path1.indir();
				p1OutDir = -path0.indir();
				path0 = path0.previous();
				path1 = path1.previous();
			}
			float p1CosOut = path1.cosine(p1OutDir);
			for(int i = merge ? pl0-1 : pl0; i > 0; --i) {
				float a0 = path0.footprint().get_area();
				float nextInCos = -path0.cosine(p1OutDir);
				f1 = f1.add_segment(p1Pdf, p1Ortho, p1H, p1Eta.inCos, p1CosOut, p1Eta.eta, p1Dist, nextInCos, 1.0f);
				float a1 = f1.get_area();
				reuseCount[i] = ivcm_heuristic(numPhotons, a0, a1);
				p1Pdf = float(path0.pdf_back());
				p1Ortho = false;
				p1Dist = path0.indist();
				p1H = path0.ext().curvature;
				p1OutDir = -path0.indir();
				p1CosOut = path0.cosine(p1OutDir);
				p1Eta = path0.eta(m_sceneDesc.media);
				path0 = path0.previous();
			}
			float p0CosOut = path0.cosine(p0OutDir);
			for(int i = pl0+1; i < pl; ++i) {
				float a1 = path1.footprint().get_area();
				float nextInCos = -path1.cosine(p0OutDir);
				f0 = f0.add_segment(p0Pdf, p0Ortho, p0H, p0Eta.inCos, p0CosOut, p0Eta.eta, p0Dist, nextInCos, 1.0f);
				float a0 = f0.get_area();
				reuseCount[i] = ivcm_heuristic(numPhotons, a0, a1);
				p0Pdf = float(path1.pdf_back());
				p0Ortho = false;
				p0Dist = path1.indist();
				p0H = path1.ext().curvature;
				p0OutDir = -path1.indir();
				p0CosOut = path1.cosine(p0OutDir);
				p0Eta = path1.eta(m_sceneDesc.media);
				path1 = path1.previous();
			}
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
