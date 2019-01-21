#pragma once

#include "int_types.hpp"
#include <cstddef>
#include <iostream>
#include <streambuf>

namespace mufflon { namespace util {

// Stream buffer to enable IOStreams on arrays
// Taken from https://stackoverflow.com/questions/7781898/get-an-istream-from-a-char
class ArrayStreamBuffer : public std::streambuf {
public:
	ArrayStreamBuffer(const char* begin, const std::size_t bytes);
	int_type underflow();
	int_type uflow();
	int_type pbackfail(int_type ch);
	std::streamsize showmanyc();
	std::streampos seekoff(std::streamoff off, std::ios_base::seekdir way,
						   std::ios_base::openmode which = std::ios_base::in | std::ios_base::out);
	std::streampos seekpos(std::streampos sp, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out);
	ArrayStreamBuffer(const ArrayStreamBuffer&) = delete;
	ArrayStreamBuffer(ArrayStreamBuffer&&) = default;
	ArrayStreamBuffer& operator=(const ArrayStreamBuffer&) = delete;
	ArrayStreamBuffer& operator=(ArrayStreamBuffer&&) = default;
	~ArrayStreamBuffer() = default;

private:
	const char* const m_begin;
	const char* const m_end;
	const char* m_current;
};

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
		m_stream(stream) {}

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

}} // namespace mufflon::util