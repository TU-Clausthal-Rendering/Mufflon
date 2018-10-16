#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <iostream>

int main(int argc, const char *argv[]) {
	OpenMesh::TriMesh_ArrayKernelT<> mesh;
	std::cout << "Hello little mufflon!" << std::endl;
	std::cout << mesh.n_vertices() << " vertices in the test mesh" << std::endl;
}