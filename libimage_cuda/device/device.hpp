#pragma once

#include "../defines.hpp"



class ByteBuffer
{
public:
    u8* data = nullptr;
    size_t capacity = 0;
    size_t size = 0;
};


namespace cuda
{
    bool device_malloc(ByteBuffer& buffer, size_t n_bytes);

    bool unified_malloc(ByteBuffer& buffer, size_t n_bytes);

    bool host_malloc(ByteBuffer& buffer, size_t n_bytes);

    bool free(ByteBuffer& buffer);

    bool host_free(ByteBuffer& buffer);


    u8* push_bytes(ByteBuffer& buffer, size_t n_bytes);

    bool pop_bytes(ByteBuffer& buffer, size_t n_bytes);


    bool memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes);

    bool memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes);


    bool no_errors(cstr label);

    bool launch_success(cstr label);


    enum class Malloc : int
    {
        Device,
        Unified,
        Host
    };


    template <typename T>
    class MemoryBuffer
    {
    private:
        T* data_ = nullptr;
        size_t capacity_ = 0;
        size_t size_ = 0;

        Malloc m_;

    public:
        MemoryBuffer(size_t n_elements, Malloc m)
        {
            m_ = m;
            auto const n_bytes = sizeof(T) * n_elements;

            ByteBuffer b{};

            bool result = false;

            switch(m_)
            {
                case Malloc::Device:
                    result = device_malloc(b, n_bytes);
                    break;
                case Malloc::Unified:
                    result = unified_malloc(b, n_bytes);
                    break;
                case Malloc::Host:
                    result = host_malloc(b, n_bytes);
                    break;
            }

            assert(result);

            if(result)
            {
                data_ = (T*)b.data;
                capacity_ = n_elements;
                size_ = 0;
            }
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


        bool pop(size_t n_elements)
        {
            assert(data_);
            assert(capacity_);
            assert(size_ <= capacity_);
            assert(n_elements <= capacity_);
            assert(n_elements <= size_);

            auto is_valid = 
                data_ &&
                capacity_ &&
                size_ <= capacity_ &&
                n_elements <= capacity_ &&
                n_elements <= size_;

            if (is_valid)
            {
                size_ -= n_elements;
                return true;
            }

            return false;
        }


        void reset()
        {
            size_ = 0;
        }


        void free()
        {          
            capacity_ = 0;
            size_ = 0;

            if(!data_)
            {
                return;
            }

            ByteBuffer b{};
            b.data = (u8*)data_;

            switch(m_)
            {
                case Malloc::Device:
                    cuda::free(b);
                    break;
                case Malloc::Unified:
                    cuda::free(b);
                    break;
                case Malloc::Host:
                    cuda::host_free(b);
                    break;
            }
        }


        size_t avail()
        {
            return capacity_ - size_;
        }
    };
    
}