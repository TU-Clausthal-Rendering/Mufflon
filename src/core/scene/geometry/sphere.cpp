#include "sphere.hpp"
#include <cuda_runtime.h>

namespace mufflon::scene::geometry {

Spheres::SphereHandle Spheres::add(const Point& point, float radius, MaterialIndex idx) {
	SphereHandle hdl(m_sphereData.get_size());
	m_attributes.resize(m_sphereData.get_size() + 1u);
	auto posRadAccessor = m_sphereData.aquire<>();
	auto matIndexAccessor = m_matIndex.aquire<>();
	posRadAccessor->back().m_radPos.position = point;
	posRadAccessor->back().m_radPos.radius = radius;
	matIndexAccessor->back() = idx;
	return hdl;
}

Spheres::BulkReturn Spheres::add_bulk(std::size_t count, std::istream& radPosStream) {
	SphereHandle hdl(m_sphereData.get_size());
	m_sphereData.read(count, radPosStream);
	std::size_t readRadPos = static_cast<std::size_t>(radPosStream.gcount()) / sizeof(Sphere);
	return {hdl, readRadPos};
}

/*template <>
const Spheres::DeviceHandles<Residency::CUDA>& Spheres::make_resident<Residency::CUDA>() {
	if(!m_cudaDirty) {
		// Nothing dirty -> no need to do anything
		mAssert(m_cudaHandles != nullptr);
		return *m_cudaHandles;
	}
	if(m_cudaHandles) {
		// Already have device pointer -> gotta free them first
		mAssert(m_cudaHandles->m_spheres != nullptr);
		mAssert(m_cudaHandles->m_matIndices != nullptr);
		mAssert(m_cudaHandles->m_attributes != nullptr);
		cudaFree(m_cudaHandles->m_spheres);
		cudaFree(m_cudaHandles->m_matIndices);
		for(u32 i = 0u; i < m_cudaHandles->m_numAttribs; ++i)
			cudaFree(m_cudaHandles->m_attributes[i]);
		cudaFree(m_cudaHandles->m_attributes);
		m_cudaHandles.reset();
	}
	
	m_cudaHandles = std::make_unique<DeviceHandles<Residency::CUDA>>(nullptr);
	// Allocate the data with CUDA
	// TODO: use custom allocator?
	mAssert(m_attributes.size() >= 2u);
	m_cudaHandles->m_numSpheres = m_sphereData.size();
	m_cudaHandles->m_numAttribs = m_attributes.size() - 2u;
	// TODO: CUDA errors?
	cudaMalloc(&m_cudaHandles->m_spheres, sizeof(Sphere) * m_sphereData.size());
	cudaMalloc(&m_cudaHandles->m_matIndices, sizeof(MaterialIndex) * m_matIndex.size());
	cudaMalloc(&m_cudaHandles->m_attributes, sizeof(void*) * m_attributes.size());
	auto iter = m_attributes.begin();
	++iter; // Pos+rad already covered
	++iter; // MaterialIndex already covered
	for(std::size_t i = 2u; i < m_attributes.size(); ++i) {
		mAssert(iter != m_attributes.end());
		cudaMalloc(&m_cudaHandles->m_attributes[i], iter->elem_size() * iter->size());
		++iter;
	}
	m_cudaDirty = false;

	return *m_cudaHandles;
}*/

} // namespace mufflon::scene::geometry