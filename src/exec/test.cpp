#include <vector>
#include <unordered_map>
#include <string>
#include <string_view>
#include <iostream>
#include <memory>
#include <algorithm>

using namespace std;
using PayLoad = int;

unique_ptr<const char[]> conv(string_view str) {
	unique_ptr<char[]> p (new char [str.size()+1]);
	memcpy(p.get(), str.data(), str.size()+1);
	return move(p);
}

int main() {
	unordered_map<string_view, PayLoad> map;
	vector<unique_ptr<const char[]>> mapKeyStore;
	// Add multiple values
	mapKeyStore.push_back(conv("a"));
	map.emplace(mapKeyStore.back().get(), 3);
	mapKeyStore.push_back(conv("b"));
	map.emplace(mapKeyStore.back().get(), 1);
	mapKeyStore.push_back(conv("c"));
	map.emplace(mapKeyStore.back().get(), 4);
	// Search all keys
	cout << map.find("a")->second;
	cout << map.find("b")->second;
	cout << map.find("c")->second;
	// Delete the "a" key
	map.erase("a");
	mapKeyStore.erase(remove_if(mapKeyStore.begin(), mapKeyStore.end(),
		[](const auto& a){ return strcmp(a.get(), "a") == 0; }),
		mapKeyStore.end());
	// Test if verything is OK.
	cout << '\n';
	for(auto it : map)
		cout << it.first << ": " << it.second << "\n";

	return 0;
}