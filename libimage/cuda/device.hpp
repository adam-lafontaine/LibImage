#pragma once

#include "../types.hpp"

#include <cstddef>
#include <cassert>
#include <array>
#include <vector>

bool cuda_memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes);

bool cuda_memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes);


bool cuda_no_errors();

bool cuda_launch_success();


template <typename T>
class DeviceArray
{
public:
    T* data = nullptr;
    u32 n_elements = 0;
};


template <class T, size_t N>
bool copy_to_device(std::array<T, N> const& src, DeviceArray<T>& dst)
{
    assert(dst.data);
    assert(dst.n_elements);
    assert(dst.n_elements == src.size());

    auto bytes = N * sizeof(T);

    return cuda_memcpy_to_device(src.data(), dst.data, bytes);
}


template <typename T>
bool copy_to_device(std::vector<T> const& src, DeviceArray<T>& dst)
{
    assert(dst.data);
    assert(dst.n_elements);
    assert(dst.n_elements == src.size());

    auto bytes = src.size() * sizeof(T);

    return cuda_memcpy_to_device(src.data(), dst.data, bytes);
}



namespace device
{
    class MemoryBuffer
    {
    public:
        u8* data = nullptr;
        size_t capacity = 0;
        size_t size = 0;
    };


    bool malloc(MemoryBuffer& buffer, size_t n_bytes);

    bool free(MemoryBuffer& buffer);

    u8* push(MemoryBuffer& buffer, size_t n_bytes);

    bool pop(MemoryBuffer& buffer, size_t n_bytes);

    bool is_valid(MemoryBuffer const& buffer);
}