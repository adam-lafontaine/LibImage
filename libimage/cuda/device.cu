/*

Copyright (c) 2021 Adam Lafontaine

*/
#include "device.hpp"
#include "cuda_def.cuh"

#include <iostream>


static void check_error(cudaError_t err)
{
    if(err == cudaSuccess)
    {
        return;
    }

    std::cout 
        << "\n*** CUDA ERROR ***\n\n" 
        << cudaGetErrorString(err)
        << "\n\n******************\n\n";
}


bool cuda_device_malloc(void** ptr, u32 n_bytes)
{
    cudaError_t err = cudaMalloc(ptr, n_bytes);
    check_error(err);
    
    return err == cudaSuccess;
}


bool cuda_device_free(void* ptr)
{
    cudaError_t err = cudaFree(ptr);
    check_error(err);

    return err == cudaSuccess;
}


bool cuda_memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes)
{
    cudaError_t err = cudaMemcpy(device_dst, host_src, n_bytes, cudaMemcpyHostToDevice);
    check_error(err);

    return err == cudaSuccess;
}


bool cuda_memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes)
{
    cudaError_t err = cudaMemcpy(host_dst, device_src, n_bytes, cudaMemcpyDeviceToHost);
    check_error(err);

    return err == cudaSuccess;
}


bool cuda_no_errors()
{
    cudaError_t err = cudaGetLastError();
    check_error(err);

    return err == cudaSuccess;
}


bool cuda_launch_success()
{
    cudaError_t err = cudaDeviceSynchronize();
    check_error(err);

    return err == cudaSuccess;
}