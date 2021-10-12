#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "../types.hpp"

#include <cstddef>

class DeviceBuffer
{
public:
    u8* data = nullptr;
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
bool push_array(DeviceArray<T>& arr, DeviceBuffer& buffer, u32 n_elements)
{
    auto bytes = n_elements * sizeof(T);
    bool result = buffer.offset + bytes <= buffer.total_bytes;

    if(result)
    {
        arr.data = (T*)(buffer.data + buffer.offset);
        arr.n_elements = n_elements;
        buffer.offset += bytes;
    }

    return result;
}


template <typename T>
void pop_array(DeviceArray<T>& arr, DeviceBuffer& buffer)
{
    auto bytes = arr.n_elements * sizeof(T);
    buffer.offset -= bytes;

    arr.data = NULL;
    arr.n_elements = 0;
}


bool device_malloc(DeviceBuffer& buffer, size_t n_bytes);

bool device_free(DeviceBuffer& buffer);


bool memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes);

bool memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes);


template <typename T>
bool copy_to_device(const void* src, DeviceArray<T> const& dst, size_t n_bytes)
{
    return memcpy_to_device(src, dst.data, n_bytes);
}


template <typename T>
bool copy_to_host(DeviceArray<T> const& src, void* dst, size_t n_bytes)
{
    return memcpy_to_host(src.data, dst, n_bytes);
}