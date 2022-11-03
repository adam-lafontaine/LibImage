#pragma once

#include <cstddef>
#include <cassert>
#include <cstdint>
#include <cstdlib>

#define LIBIMAGE_PNG
#define LIBIMAGE_BMP

//#define LIBIMAGE_NO_WRITE
//#define LIBIMAGE_NO_RESIZE
//#define LIBIMAGE_NO_PARALLEL
//#define LIBIMAGE_NO_FILESYSTEM

//#define LIBIMAGE_NO_CPP17

//#define RPI_3B_PLUS
#define JETSON_NANO


#ifdef RPI_3B_PLUS

#define LIBIMAGE_NO_CPP17
#define SIMD_ARM_NEON

#endif // RPI_3B_PLUS

#ifdef JETSON_NANO

#define LIBIMAGE_NO_CPP17
#define SIMD_ARM_NEON

#endif // JETSON_NANO


#ifdef LIBIMAGE_NO_CPP17

#define LIBIMAGE_NO_PARALLEL
#define LIBIMAGE_NO_FILESYSTEM

#endif // LIBIMAGE_NO_CPP17


/*  types.hpp  */

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using r32 = float;
using r64 = double;
using i32 = int32_t;
using cstr = const char*;


#ifdef LIBIMAGE_NO_PARALLEL

constexpr u32 N_THREADS = 1;

#else

constexpr u32 N_THREADS = 8;

#endif //!LIBIMAGE_NO_PARALLEL


class Point2Du32
{
public:
	u32 x;
	u32 y;
};


class Point2Dr32
{
public:
	r32 x;
	r32 y;
};


// region of interest in an image
class Range2Du32
{
public:
	u32 x_begin;
	u32 x_end;   // one past last x
	u32 y_begin;
	u32 y_end;   // one past last y
};


template<typename T>
class MemoryBuffer
{
public:
    T* data = nullptr;
    size_t capacity = 0;
    size_t size = 0;    
};

/*
template <typename T>
class MemoryBuffer
{
private:
	T* data_ = nullptr;
	size_t capacity_ = 0;
	size_t size_ = 0;

public:
	MemoryBuffer(size_t n_elements)
	{
		auto data = std::malloc(sizeof(T) * n_elements);
		assert(data);

		data_ = (T*)data;
		capacity_ = n_elements;
	}


	T* push(size_t n_elements)
	{
		assert(data_);
		assert(capacity_);
		assert(size_ < capacity_);

		auto is_valid =
			data_ &&
			capacity_ &&
			size_ < capacity_;

		auto elements_available = (capacity_ - size_) >= n_elements;
		assert(elements_available);

		if (!is_valid || !elements_available)
		{
			return nullptr;
		}

		auto data = data_ + size_;

		size_ += n_elements;

		return data;
	}


	void reset()
	{
		size_ = 0;
	}


	void free()
	{
		if (data_)
		{
			std::free(data_);
			data_ = nullptr;
		}

		capacity_ = 0;
		size_ = 0;
	}
};*/