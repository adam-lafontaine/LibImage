
#include "../libimage/libimage.hpp"

#include <cstdio>

namespace fs = std::filesystem;
namespace img = libimage;

constexpr auto ROOT_DIR = "~/Repos/LibImage/CudaTests";

const auto ROOT_PATH = fs::path(ROOT_DIR);

// make sure these files exist
const auto CORVETTE_PATH = ROOT_PATH / "in_files/corvette.png";
const auto CADILLAC_PATH = ROOT_PATH / "in_files/cadillac.png";

const auto DST_IMAGE_ROOT = ROOT_PATH / "out_files";

int main()
{
    printf("hello from main\n");

    img::image_t src;
    img::read_image_from_file(CORVETTE_PATH, src);

    img::write_image(src, DST_IMAGE_ROOT / "corvette_out.png");
}