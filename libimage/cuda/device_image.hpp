#include "../rgba.hpp"
#include "../gray.hpp"



namespace libimage
{



    namespace cuda
    {

        class DeviceRGBAImage
        {
        public:
            u32 width;
            u32 height;

            pixel_t* data;
        };


        class DeviceGrayImage
        {
        public:
            u32 width;
            u32 height;

            u8* data;
        };


    }
}