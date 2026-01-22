#include <iostream>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

void dome_2d_layout_example() {
    printf("\n2D Layout example:\n");


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

void dome_4d_layout_example() {
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

    using shape = Shape<Shape<_2, _3>, Shape<_2, _3>>;
    using stride = Stride<Stride<Int<18>, _3>, Stride<Int<9>, _1>>;

    Tensor t = make_tensor(&data[0], Layout<shape, stride>{});

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

void product_example() {
    printf("\nProduct Layout example:\n");

    using layout1 = Layout<Shape<_2, _3>, Stride<_3, _1>>;
    using layout2 = Layout<Shape<_4, _5>, Stride<_5, _1>>;

    using prod_layout = decltype(logical_product(layout1{}, layout2{}));
    print_layout(prod_layout{});
}

int main() {
    slice_example();

    return 0;
}
