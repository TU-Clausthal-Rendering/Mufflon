#include <vector>
#include <unordered_map>
#include <string>
#include <string_view>
#include <iostream>
#include <memory>
#include <algorithm>

#include <cuda_runtime.h>
#include "core/scene/geometry/polygon.hpp"

using namespace mufflon::scene;

class VectorStream : public std::basic_streambuf<char, std::char_traits<char>> {
public:
	template < class U >
	VectorStream(std::vector<U>& vec) {
		this->setg(reinterpret_cast<char*>(vec.data()), reinterpret_cast<char*>(vec.data()),
				   reinterpret_cast<char*>(vec.data() + vec.size()));
	}
};

void test_polygon() {
	using namespace geometry;

	Polygons poly;
	auto impHandle = poly.request<Polygons::VertexAttributeHandle<float>>("importance");
	auto indexHandle = poly.request<Polygons::FaceAttributeHandle<unsigned int>>("index");

	auto v0 = poly.add(Point(0, 0, 0), Normal(1, 0, 0), UvCoordinate(0, 0));
	auto v1 = poly.add(Point(1, 0, 0), Normal(1, 0, 0), UvCoordinate(1, 0));
	auto v2 = poly.add(Point(0, 1, 0), Normal(1, 0, 0), UvCoordinate(0, 1));
	auto f0 = poly.add(v0, v1, v2, 0u);
	std::vector<Point> points{ Point(0, 0, 1), Point(1, 0, 1), Point(0, 1, 1), Point(1, 1, 1) };
	std::vector<Point> normals{ Normal(-1, -1, 0), Normal(1, -1, 0), Normal(-1, 1, 1), Normal(1, 1, 1) };
	std::vector<UvCoordinate> uvs{ UvCoordinate(0, 0), UvCoordinate(1, 0), UvCoordinate(0, 1), UvCoordinate(1, 1) };
	std::vector<MaterialIndex> mats{ 5u };

	VectorStream pointBuffer(points);
	VectorStream normalBuffer(normals);
	VectorStream uvBuffer(uvs);
	VectorStream matBuffer(mats);
	std::istream pointStream(&pointBuffer);
	std::istream normalStream(&normalBuffer);
	std::istream uvStream(&uvBuffer);
	std::istream matStream(&matBuffer);

	auto f1 = poly.add_bulk(points.size(), pointStream, normalStream, uvStream);
	poly.add_bulk<>(poly.get_mat_indices(), f1.handle, mats.size(), matStream);
}



int main() {
	test_polygon();
	return 0;
}