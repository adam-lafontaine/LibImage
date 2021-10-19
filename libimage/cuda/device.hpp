#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "../types.hpp"

#include <cstddef>
#include <cassert>


bool cuda_device_malloc(void** ptr, u32 n_bytes);

bool cuda_device_free(void* ptr);


bool cuda_memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes);

bool cuda_memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes);


bool cuda_no_errors();

bool cuda_launch_success();



template <typename T>
class DeviceBuffer
{
public:
    T* data = nullptr;
    u32 total_bytes = 0;
    u32 offset = 0;
};


template <typename T>
class DeviceArray
{
public:
    T* data = nullptr;
    u32 n_elements = 0;
};


template <typename T>
bool push_array(DeviceArray<T>& arr, DeviceBuffer<T>& buffer, u32 n_elements)
{
    auto bytes = n_elements * sizeof(T);
    bool result = buffer.offset + bytes <= buffer.total_bytes;

    if(result)
    {
        arr.data = (T*)((u8*)buffer.data + buffer.offset);
        arr.n_elements = n_elements;
        buffer.offset += bytes;
    }

    return result;
}


template <typename T>
void pop_array(DeviceArray<T>& arr, DeviceBuffer<T>& buffer)
{
    auto bytes = arr.n_elements * sizeof(T);
    buffer.offset -= bytes;

    arr.data = NULL;
    arr.n_elements = 0;
}


template <typename T>
bool device_malloc(DeviceBuffer<T>& buffer, size_t n_bytes)
{
    bool result = cuda_device_malloc((void**)&(buffer.data), n_bytes);
    if(result)
    {
        buffer.total_bytes = n_bytes;
    }

    return result;
}


template <typename T>
bool device_free(DeviceBuffer<T>& buffer)
{
    buffer.total_bytes = 0;
    buffer.offset = 0;
    return cuda_device_free(buffer.data);
}


template <typename T>
bool memcpy_to_device(const void* src, DeviceArray<T> const& dst)
{
    assert(src);
    assert(dst.data);
    assert(dst.n_elements);

    auto bytes = dst.n_elements * sizeof(T);
    return cuda_memcpy_to_device(src, dst.data, bytes);
}


template <typename T>
bool memcpy_to_host(DeviceArray<T> const& src, void* dst)
{
    assert(src.data);
    assert(src.n_elements);
    assert(dst);

    auto bytes = src.n_elements * sizeof(T);
    return cuda_memcpy_to_host(src.data, dst, bytes);
}
