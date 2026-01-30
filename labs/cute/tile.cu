#include <cute/tensor.hpp>
#include <array>

using namespace cute;


void local_tile_example() {
    using namespace cute;

    printf("\nTiling tensor example:\n");


    auto A = make_tensor<int>(make_layout(Shape<_6, _8>{}, Stride<_8, _1>{}));
    for (int i = 0; i < size(A); i++) {
        A[i] = i;
    }
    print_tensor(A);

    auto tiler = Shape<_4, _4>{};
    auto coord = make_coord(1, 1);

    Tensor tile = local_tile(A, tiler, coord);
    print_tensor(tile);
}


int main() {
    local_tile_example();

    return 0;
}
