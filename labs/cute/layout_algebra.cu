#include <iostream>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

void coalesce_example() {
    Layout<Shape<_3, _4>, Stride<_1, _3>> layout;
    
    auto layout2 = coalesce(layout);
    printf("\n");
    print(layout);

    printf("\n");
    print(layout2);

    printf("\n");
    print_layout(layout);

    printf("\n");
    print(coalesce(Layout<Shape<_2, Shape<_2, Shape<_2, Shape<_2, Shape<_2, Shape<_2, _4>>>>>>>{}, Step<_1, Step<_1, _1>>{}));

    printf("\n");
}


void composition_example_1() {
    std::array<int, 24> data;
    for (int i = 0; i < 24; ++i) {
        data[i] = i;
    }

    Layout<Shape<_4, _6>, Stride<_6, _1>> a;
    Tensor t = make_tensor(data.data(), a);
    print_tensor(t);

    print_layout(a);

    auto tiler = make_tile(Layout<_2, _2>{},  // Apply 3:4 to mode-0
                           Layout<_3, _2>{});   // Apply 8:2 to mode-1
    
    auto b = composition(a, tiler);
    print_layout(b);


    Tensor t_a = make_tensor(data.data(), a);
    Tensor t_b = make_tensor(data.data(), b);

    print_tensor(t_a);
    print_tensor(t_b);

}

void complement_example() {
    std::array<int, 128> data;
    for (int i = 0; i < 128; ++i) {
        data[i] = i;
    }

    Layout<Shape<_2, _3>, Stride<_9, _1>> a;
    Tensor t = make_tensor(data.data(), a);

    auto b = complement(a, Int<54>{});
    print_layout(a);
    print_layout(b);
    auto c = make_layout(a, b);
    print(c); printf("\n");
      print_tensor(t); printf("\n");
    Tensor t_c = make_tensor(data.data(), c);
    print_tensor(t_c); printf("\n");
}

void complement_example2() {
    std::array<int, 128> data;
    for (int i = 0; i < 128; ++i) {
        data[i] = i;
    }

    Layout<Shape<_2, _3>, Stride<Int<18>, _2>> a;
    Tensor t = make_tensor(data.data(), a);

    auto b = complement(a, Int<72>{});
    print(a); printf("\n");
    print(b); printf("\n");
    auto c = make_layout(a, b);
    print(c); printf("\n");

    // print_tensor(t); printf("\n");
    // Tensor t_c = make_tensor(data.data(), c);
    // print_tensor(t_c); printf("\n");
}

void divide_example() {

    Layout<Shape<_16>, Stride<_1>> a;
    Layout<Shape<_4>, Stride<_2>> b;

    auto c = logical_divide(a, b);
    print(a);
    printf("\n");
    print(b);
    printf("\n");

    print(c);
    printf("\n");

}

void product_example() {
    Layout<Shape<_2, _2>, Stride<Int<16>, _4>> a;
    Layout<Shape<_8>, Stride<_1>> b;

    auto c = logical_product(a, b);
    print(a);
    printf("\n");
    print(b);
    printf("\n");

    print(c);
    printf("\n");


    std::array<int, 128> data;
    for (int i = 0; i < 128; ++i) {
        data[i] = i;
    }
    Tensor ta = make_tensor(data.data(), a);
    print_tensor(ta);

    Tensor t = make_tensor(data.data(), c);
    print_tensor(t);


}

void product_example2() {
    printf("\nProduct Layout example:\n");

    using layout1 = Layout<Shape<_2, _3>, Stride<_3, _1>>;
    using layout2 = Layout<Shape<_4, _5>, Stride<_5, _1>>;

    using prod_layout = decltype(logical_product(layout1{}, layout2{}));
    print_layout(prod_layout{});
}

int main() {
    // complement_example();
    // complement_example2();
    product_example();
    return 0;
}
