#include "attribute_list.hpp"


using namespace mufflon::scene;

template OmAttributePool<true>;
template OmAttributePool<false>;

void test(AttributeList<>&& pool) {
	auto x = pool.add<int>("test");
	auto& attr = pool.aquire(x);
	int* i = *attr.aquire<>();
	pool.remove(x);
}

void test2(OmAttributeList<true>&& pool) {
	OpenMesh::FPropHandleT<float> hdl;
	auto x = pool.add<float>("test", hdl);
	auto& attr = pool.aquire(x);
	float* f = *attr.aquire<>();
	pool.remove(x);
}

void test3(OmAttributeList<false>&& pool) {
	OpenMesh::VPropHandleT<float> hdl;
	auto x = pool.add<float>("test", hdl);
	auto& attr = pool.aquire(x);
	float* f = *attr.aquire<>();
	pool.remove(x);
}

void test() {
	{
		geometry::PolygonMeshType mesh;
		OmAttributeList<true> pool{ mesh };
		std::map<char, OmAttributeList<true>> map;
		//map.emplace('c', std::move(pool));
		static_assert(std::is_convertible_v<OmAttributeList<true>, OmAttributeList<true>>,
					  "Why not?");
	}
	{
		geometry::PolygonMeshType mesh;
		OmAttributeList<false> pool{ mesh };
		test3(std::move(pool));
	}
	{
		AttributeList pool;
		test(std::move(pool));
	}
}