#pragma once

#include <cstddef>
#include <iostream>

namespace mufflon::util {

// Interface abstracting C++ iostream/FILE descriptor into common type
class IByteReader {
public:
	virtual std::size_t read(char* mem, std::size_t bytes) = 0;
};
class IByteWriter {
public:
	virtual std::size_t write(const char* mem, std::size_t bytes) = 0;
};
class IByteIO : public IByteReader, public IByteWriter {};

// Implementation for C++ streams
class StreamReader : public IByteReader {
public:
	StreamReader(std::istream& stream) :
		m_stream(stream)
	{}

	virtual std::size_t read(char* mem, std::size_t bytes) override {
		m_stream.read(mem, bytes);
		return m_stream.gcount();
	}

private:
	std::istream& m_stream;
};
class StreamWriter : public IByteWriter {
public:
	StreamWriter(std::ostream& stream) :
		m_stream(stream) {}

	virtual std::size_t write(const char* mem, std::size_t bytes) override {
		m_stream.write(mem, bytes);
		return bytes;
	}

private:
	std::ostream& m_stream;
};
class StreamIO : public IByteIO {
public:
	StreamIO(std::iostream& stream) :
		m_stream(stream) {}

	virtual std::size_t read(char* mem, std::size_t bytes) override {
		m_stream.read(mem, bytes);
		return m_stream.gcount();
	}

	virtual std::size_t write(const char* mem, std::size_t bytes) override {
		m_stream.write(mem, bytes);
		return bytes;
	}
private:
	std::iostream& m_stream;
};

// Implementation for C FILE descriptors
class FileReader : public IByteReader {
public:
	FileReader(std::FILE* desc) :
		m_descriptor(desc) {}

	virtual std::size_t read(char* mem, std::size_t bytes) override {
		return std::fread(mem, 1u, bytes, m_descriptor);
	}

private:
	std::FILE* m_descriptor;
};
class FileWriter : public IByteWriter {
public:
	FileWriter(std::FILE* desc) :
		m_descriptor(desc) {}

	virtual std::size_t write(const char* mem, std::size_t bytes) override {
		return std::fwrite(mem, 1u, bytes, m_descriptor);
	}

private:
	std::FILE* m_descriptor;
};
class FileIO : public IByteIO {
public:
	FileIO(std::FILE* desc) :
		m_descriptor(desc) {}

	virtual std::size_t read(char* mem, std::size_t bytes) override {
		return std::fread(mem, 1u, bytes, m_descriptor);
	}

	virtual std::size_t write(const char* mem, std::size_t bytes) override {
		return std::fwrite(mem, 1u, bytes, m_descriptor);
	}

private:
	std::FILE* m_descriptor;
};

} // namespace mufflon::util