#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <iostream>

namespace mesh {

	void test() {
		OpenMesh::TriMesh_ArrayKernelT<> mesh;
		std::cout << "Hello little mufflon!" << std::endl;
		std::cout << mesh.n_vertices() << " vertices in the test mesh" << std::endl;
	}

} // namespace mesh