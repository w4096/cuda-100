#include <iostream>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

void layout_2d_example() {
    printf("2D Layout example:\n");


    Layout<Shape<_2, _4>, Stride<_4, _1>> layout;
    Layout<Shape<_4, _2>, Stride<_1, _4>> layout_t;

    int data[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    Tensor a = make_tensor(&data[0], layout);

    Tensor a_t = make_tensor(&data[0], layout_t);


    print_layout(layout);

    print_layout(layout_t);


    print_tensor(a);
    print_tensor(a_t);
}

void layout_4d_example() {
    printf("\n2D Layout example:\n");

    constexpr int N = 120;
    int data[N] = {};
    for (int i = 0; i < N; ++i) {
        data[i] = i;
    }

    using shape = Shape<Shape<_4, _5>, Shape<_2, _3>>;
    using stride = Stride<Stride<Int<30>, _3>, Stride<Int<15>, _1>>;

    Tensor b = make_tensor(&data[0], Layout<shape, stride>{});
    for (int i = 0; i < size<0, 0>(b); i++) {
        for (int j = 0; j < size<0, 1>(b); j++) {
            printf("Block (%d, %d):\n", i, j);
            print_tensor(b(make_coord(i, j), make_coord(_, _)));
        }
    }
}

void slice_example() {
    constexpr int N = 36;
    int data[N] = {};
    for (int i = 0; i < N; ++i) {
        data[i] = i;
    }

    using shape = Shape<Shape<_2, _2>, Shape<_3, _3>>;
    using stride = Stride<Stride<_3, _6>, Stride<_1, _12>>;

    Tensor t = make_tensor(&data[0], Layout<shape, stride>{});

    print_tensor(t);


    printf("\ncoord: (_, _), (0, 0):\n");
    print_tensor(t(make_coord(_, _), make_coord(0, 0)));

    printf("\ncoord: (1, 1), (_, _):\n");
    print_tensor(t(make_coord(1, 1), make_coord(_, _)));

    printf("\ncoord: (_, 1), (_, 1):\n");
    print_tensor(t(make_coord(_, 1), make_coord(_, 1)));

    printf("\ncoord: (1, _), (1, _):\n");
    print_tensor(t(make_coord(1, _), make_coord(1, _)));

    printf("\ncoord: (_, 0), (1, 1):\n");
    print_tensor(t(make_coord(_, 0), make_coord(1, 1)));

    printf("\ncoord: (0, _), (1, 1):\n");
    print_tensor(t(make_coord(0, _), make_coord(1, 1)));
}

int main() {
    slice_example();
    return 0;
}
