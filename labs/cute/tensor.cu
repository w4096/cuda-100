#include <cute/tensor.hpp>
#include <array>

using namespace cute;

void tensor_example() {


    std::array<int, 8> data;
    for (int i = 0; i < data.size(); i++) {
        data[i] = i;
    }

    Layout<Shape<_2, _4>, Stride<_4, _1>> layout;
    Tensor t = make_tensor(make_gmem_ptr<int>(data.data()), layout);
    print_tensor(t);

    for (int j = 0; j < size<0>(t); j++) {
        for (int k = 0; k < size<1>(t); k++) {
            printf("%2d ", t(j, k));
        }
        printf("\n");
    }
}

void create_non_owning_tensor_example() {
    using namespace cute;

    std::array<int, 128> data;
    for (int i = 0; i < 128; i++) {
        data[i] = i;
    }

    // static layout

    // shape: (8, 16), stride: (16, 1)
    Tensor t = make_tensor(make_gmem_ptr<int>(data.data()), make_layout(make_shape(Int<8>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{})));
    print_tensor(t);

    Tensor t2 = make_tensor(make_gmem_ptr(data.data()), make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
    print_tensor(t2);


    // dynamic layout
    Tensor t3 = make_tensor(make_gmem_ptr(data.data()), make_shape(8, 16), make_stride(16, 1));
    print_tensor(t3);
}

void create_owning_tensor_example() {

    Layout<Shape<_8, _16>, Stride<_16, _1>> layout;

    Tensor t = make_tensor<int>(layout);

    printf("%zu bytes allocated for tensor storage\n", sizeof(t));

    Tensor tv = make_tensor(t.data(), make_shape(Int<128>{}));
    for (int i = 0; i < 128; i++) {
        tv[i] = i;
    }
    print_tensor(t);
}

void tensor_slice_example() {
    std::array<int, 164> data;
    for (int i = 0; i < 164; i++) {
        data[i] = i;
    }
    int *ptr = data.data();

    // ((_3,2),(2,_5,_2)):((4,1),(_2,13,100)):
    Tensor A = make_tensor(ptr, make_shape(make_shape(Int<3>{},2), make_shape (2, Int<5>{}, Int<2>{})), make_stride(make_stride(4,1), make_stride(Int<2>{}, 13, 100)));


    // ((2,_5,_2)):((_2,13,100))
    Tensor B = A(2,_);

    // ((_3,_2)):((4,1))
    Tensor C = A(_,5);

    // (_3,2):(4,1)
    Tensor D = A(make_coord(_,_),5);

    // (_3,_5):(4,13)
    Tensor E = A(make_coord(_,1),make_coord(0,_,1));

    // (2,2,_2):(1,_2,100)
    Tensor F = A(make_coord(2,_),make_coord(_,3,_));

    Tensor G = A(make_coord(_,_), make_coord(_,_,_));

    print(A.layout()); printf("\n");
    print(B.layout()); printf("\n");
    print(C.layout()); printf("\n");
    print(D.layout()); printf("\n");
    print(E.layout()); printf("\n");
    print(F.layout()); printf("\n");
    print(G.layout()); printf("\n");

    printf("rank B: %d  rank C: %d  rank D: %d  rank E: %d  rank F: %d  rank G: %d\n", B.layout().rank, C.layout().rank, D.layout().rank, E.layout().rank, F.layout().rank, G.layout().rank);


    {
        constexpr int N = 36;
int data[N] = {};
for (int i = 0; i < N; ++i) {
   data[i] = i;
}

using shape = Shape<Shape<_2, _3>, Shape<_2, _3>>;
using stride = Stride<Stride<_6, _12>, Stride<_3, _1>>;

Tensor t = make_tensor(&data[0], Layout<shape, stride>{});
print_tensor(t);
    }


    {
        int N = 4 * 8;
        int data[N] = {};
        for (int i = 0; i < N; ++i) {
            data[i] = i;
        }
        int *ptr = &data[0];
        Tensor A = make_tensor(ptr, Layout<Shape<_4, _8>, Stride<_8, _1>>{});  // (8,24)
        auto tiler = Shape<_2,_4>{};                    // (_4,_8)

        Tensor tiled_a = zipped_divide(A, tiler);       // ((_4,_8),(2,3))

        print(A.layout());
        print(tiled_a.layout());

        print_tensor(A);
        print_tensor(tiled_a(make_coord(_, _), make_coord(0,0)));
    }

}

void slice_tensor_example() {
    using namespace cute;

    printf("\nTilling tensor example:\n");

    Tensor t = make_tensor<int>(make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})));
    for (int i = 0; i < size<0>(t); i++) {
        auto row = t(i, _);
        for (int k = 0; k < size<0>(row); k++) {
            row(k) = i * size<1>(t) + k;
        }
    }
    print_tensor(t);

    Tensor t1 = t(2, _); // row 2
    print_tensor(t1);

    Tensor t2 = t(_, 4); // column 4
    print_tensor(t2);

    Tensor t3 = make_tensor(t.data(), make_shape(make_shape(2, 2), make_shape(2, 4)), make_stride(make_stride(16, 8), make_stride(4, 1)));
    printf("\nSub-tensor t3 layout:\n");
    print(t3.layout());
    printf("\ntensor t3 data:\n");
    print_tensor(t3);

    printf("\nt4:\n");
    auto t4 = t3(make_coord(1, 1), make_coord(_, _));
    printf("\nt4 layout:\n");
    print_layout(t4.layout());
    printf("\nt4 data:\n");
    print_tensor(t4); // print sub-tensor
}


void partitioning() {
    using namespace cute;

    printf("\nPartitioning tensor example:\n");

    Tensor t = make_tensor<int>(make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})));
    for (int i = 0; i < size<0>(t); i++) {
        auto row = t(i, _);
        for (int k = 0; k < size<0>(row); k++) {
            row(k) = i * size<1>(t) + k;
        }
    }
    print_tensor(t);

    Tensor t2 = zipped_divide(t, Shape<_2, _4>{});
    printf("\nt2 layout:\n");
    print_layout(t2.layout());

    Tensor t3 = t2(0, make_coord(_, _));
    printf("\nt3 layout:\n");
    print_layout(t3.layout());
}

void composition_example() {
    using namespace cute;

    printf("\nComposition tensor example:\n");

    auto layout = make_layout(Shape<_4, _6>{}, LayoutRight{});
    std::array<int, cosize(layout)> data;
    for (int i = 0; i < cosize(layout); i++) {
        data[i] = i;
    }
    Tensor t = make_tensor(make_gmem_ptr<int>(data.data()), layout);

    print_tensor(t);

    Tensor t2 = composition(t, make_layout(make_shape(_3{}, _8{}), make_stride(_8{}, _1{})));
    print_tensor(t2);
    for (int i = 0; i < size<0>(t2); i++) {
        for (int j = 0; j < size<1>(t2); j++) {
            printf("%2d ", t2(i, j));
        }
        printf("\n");
    }
}

void tile_example() {
    using namespace cute;

    printf("\nTiling tensor example:\n");

    constexpr int M = 16;
    constexpr int N = 12;
    constexpr int K = 8;

    auto shape = make_shape(Int<M>{}, Int<N>{});

    auto A = make_tensor<int>(make_layout(shape, make_stride(Int<N>{}, Int<1>{})));
    for (int i = 0; i < size(A); i++) {
        A[i] = i;
    }
    print_tensor(A);

    auto cta_tiler = make_shape(_4{}, _3{});  // (BLK_M, BLK_N, BLK_K)
    auto cta_coord = make_coord(3, _);

    Tensor gA = local_tile(A, cta_tiler, cta_coord);

    printf("\ngA data:\n");
    print_tensor(gA(make_coord(_, _, 0)));

}


int main() {
    // layout_4d_example();
    // create_non_owning_tensor_example();
    //
    // create_owning_tensor_example();
    //
    // slice_tensor_example();
    //
    // partitioning();

    // composition_example();

    tensor_slice_example();

    return 0;
}
