/*

Copyright (c) 2021 Adam Lafontaine

*/
#include "device.hpp"
#include "cuda_def.cuh"

#include <cstdio>


static void check_error(cudaError_t err)
{
    if(err == cudaSuccess)
    {
        return;
    }

    printf("\n*** CUDA ERROR ***\n\n");
    printf(cudaGetErrorString(err));
    printf("\n\n******************\n\n");
}




static bool device_malloc(void** ptr, u32 n_bytes)
{
    cudaError_t err = cudaMalloc(ptr, n_bytes);
    check_error(err);
    
    return err == cudaSuccess;
}


static bool device_free(void* ptr)
{
    cudaError_t err = cudaFree(ptr);
    check_error(err);

    return err == cudaSuccess;
}


bool device_malloc(DeviceBuffer& buffer, size_t n_bytes)
{
    bool result = device_malloc((void**)&(buffer.data), n_bytes);
    if(result)
    {
        buffer.total_bytes = n_bytes;
    }

    return result;
}


bool device_free(DeviceBuffer& buffer)
{
    buffer.total_bytes = 0;
    buffer.offset = 0;
    return device_free(buffer.data);
}


bool memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes)
{
    cudaError_t err = cudaMemcpy(device_dst, host_src, n_bytes, cudaMemcpyHostToDevice);
    check_error(err);

    return err == cudaSuccess;
}


bool memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes)
{
    cudaError_t err = cudaMemcpy(host_dst, device_src, n_bytes, cudaMemcpyDeviceToHost);
    check_error(err);

    return err == cudaSuccess;
}