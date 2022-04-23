#include "device.hpp"
#include "cuda_def.cuh"

#ifdef CUDA_PRINT_ERROR

#include <cstdio>
#include <cassert>

#endif


static void check_error(cudaError_t err)
{
    if(err == cudaSuccess)
    {
        return;
    }

    #ifdef CUDA_PRINT_ERROR

    printf("\n*** CUDA ERROR ***\n\n");
    printf("%s", cudaGetErrorString(err));
    printf("\n\n******************\n\n");

    #endif
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


bool device_malloc(DeviceBuffer& buffer, size_t n_bytes)
{
    bool result = cuda_device_malloc((void**)&(buffer.data), n_bytes);
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
    return cuda_device_free(buffer.data);
}


namespace device
{
    bool malloc(MemoryBuffer& buffer, size_t n_bytes)
    {
        assert(!buffer.data);

        buffer.offset = 0;

        cudaError_t err = cudaMalloc((void**)&(buffer.data), n_bytes);
        check_error(err);

        bool result = err == cudaSuccess;

        if(result)
        {
            buffer.capacity = n_bytes;
        }
        
        return result;
    }


    bool free(MemoryBuffer& buffer)
    {
        buffer.capacity = 0;
        buffer.offset = 0;

        cudaError_t err = cudaFree(buffer.data);
        check_error(err);

        return err == cudaSuccess;
    }


    u8* push(MemoryBuffer& buffer, size_t n_bytes)
    {
        assert(is_valid(buffer));

        auto bytes_available = buffer.capacity - buffer.offset;
        assert(bytes_available >= n_bytes);

        if(!is_valid(buffer) || n_bytes > bytes_available)
        {
            return nullptr;
        }

        buffer.offset += n_bytes;

        return buffer.data + buffer.offset;
    }


    bool is_valid(MemoryBuffer const& buffer)
    {
        return 
            buffer.data &&
            buffer.capacity &&
            buffer.offset < buffer.capacity;
    }
}