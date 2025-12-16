#include <array>
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace ct {

template <size_t s0, size_t s1>
class Stride {
public:
    static constexpr size_t rank = 2;
    static constexpr size_t _0 = s0;
    static constexpr size_t _1 = s1;

    using transpose_t = Stride<s1, s0>;
};


template <size_t s0, size_t s1>
class Shape {
public:
    static constexpr size_t rank = 2;
    static constexpr size_t size = s0 * s1;
    static constexpr size_t _0 = s0;
    static constexpr size_t _1 = s1;

    using transpose_t = Shape<s1, s0>;
    using stride_t = Stride<s1, 1>;
};

struct OneLayout {
    static constexpr size_t stride = 1;
};


template <typename Shape, typename Stride=typename Shape::stride_t, typename layout=OneLayout>
struct Layout {
    using shape_t = Shape;
    using stride_t = Stride;
    using layout_t = layout;

    static constexpr size_t size = Shape::size;
    static constexpr size_t rank = Shape::rank;

    static constexpr size_t stride = Stride::_0 * Shape::_0 * layout_t::stride;


    struct transpose {
        using type = Layout<typename Shape::transpose_t, typename Stride::transpose_t>;
    };
    using transpose_t = transpose::type;

    __host__ __device__
     static constexpr size_t offset(size_t index) {
        const size_t y = index / shape_t::_1;
        const size_t x = index % shape_t::_1;
        return offset(y, x);
    }

    __host__ __device__
     static constexpr size_t offset(size_t y, size_t x) {
        if constexpr (std::is_same_v<layout_t, OneLayout>) {
            return y * stride_t::_0 + x * stride_t::_1;
        } else {
            return layout_t::offset(y * stride_t::_0 + x * stride_t::_1);
        }
    }
};

template <typename ElementType, typename Layout>
class Tile {
public:
    using layout_t = Layout;
    using element_t = ElementType;

    static constexpr size_t bytes = sizeof(element_t) * layout_t::size;

    __host__ __device__
    constexpr explicit Tile(element_t* data) : ptr_(data) {
        assert(ptr_ != nullptr && "Tile pointer cannot be null");
    }

    __host__ __device__
    element_t& operator()(size_t y, size_t x) {
        return ptr_[layout_t::offset(y, x)];
    }

    __host__ __device__
    element_t at(size_t i) const {
        assert(i < layout_t::size && "Index out of range");
        return ptr_[layout_t::offset(i)];
    }

    __host__ __device__
    element_t& at(size_t i) {
        assert(i < layout_t::size && "Index out of range");
        return ptr_[layout_t::offset(i)];
    }


    __host__ __device__
    void advance(size_t i) {
        ptr_ += i * layout_t::stride;
    }

    __host__ __device__
    void advance() {
        ptr_ += layout_t::stride;
    }

    __host__ __device__ constexpr static const size_t& size() {
        return layout_t::size;
    }

    template <size_t dim>
    __host__ __device__ constexpr static size_t size() {
        static_assert(dim < layout_t::rank, "Dimension out of range");
        if constexpr (dim == 0) {
            return layout_t::shape_t::_0;
        } else {
            return layout_t::shape_t::_1;
        }
    }

    __host__ __device__ size_t size(const size_t dim) const {
        if (dim == 0) {
            return layout_t::shape_t::_0;
        } else {
            return layout_t::shape_t::_1;
        }
    }

    template <size_t dim>
    __host__ __device__ constexpr static size_t stride() {
        static_assert(dim < layout_t::rank, "Dimension out of range");
        if constexpr (dim == 0) {
            return layout_t::stride_t::_0;
        } else {
            return layout_t::stride_t::_1;
        }
    }

    __host__ __device__ constexpr static size_t stride(const size_t dim) {
        if (dim == 0) {
            return layout_t::stride_t::_0;
        } else {
            return layout_t::stride_t::_1;
        }
    }


    __host__ __device__ element_t* data() {
        return ptr_;
    }

    __host__ __device__ const element_t* data() const {
        return ptr_;
    }

    template <typename TileA>
    __host__ __device__
    void operator=(const TileA& rhs) {
        static_assert(size() == TileA::size(), "Tile sizes must match");
        for (size_t i = 0; i < size(); ++i) {
            at(i) = rhs.at(i);
        }
    }

    template <size_t dim1, size_t dim2>
    __host__ __device__
    auto transpose() const {
        return Tile<element_t, typename layout_t::template transpose_t<dim1, dim2>>(ptr_);
    }

private:
    __host__ __device__
    element_t& index(size_t index) const {
        return ptr_[layout_t::offset(index)];
    }

    element_t* ptr_; // 数据指针
};

template <typename element_t, size_t s0, size_t s1, typename layout_t>
auto make_tile(element_t* ptr, Shape<s0, s1>, layout_t) {
    using shape_t = Shape<s0, s1>;
    using layout2_t = Layout<shape_t, typename shape_t::stride_t, layout_t>;
    return Tile<element_t, layout2_t>(ptr);
}

template <typename element_t, size_t s0, size_t s1, size_t s3, size_t s4, typename layout_t>
auto make_tile(element_t* ptr, Shape<s0, s1>, Stride<s3, s4>, layout_t) {
    using layout2_t = Layout<Shape<s0, s1>, Stride<s3, s4>, layout_t>;
    return Tile<element_t, layout2_t>(ptr);
}

template <typename element_t, size_t s0, size_t s1, size_t s3, size_t s4>
auto make_tile(element_t* ptr, Shape<s0, s1>, Stride<s3, s4>) {
    using layout_t = Layout<Shape<s0, s1>, Stride<s3, s4>>;
    return Tile<element_t, layout_t>(ptr);
}

template <typename element_t, size_t s0, size_t s1>
auto make_tile(element_t* ptr, Shape<s0, s1>) {
    using layout_t = Layout<Shape<s0, s1>>;
    return Tile<element_t, layout_t>(ptr);
}


} // namespace ct

#include <iostream>

__global__ void print(int *a) {
    ct::Tile<int, ct::Layout<ct::Shape<6, 6>>> tile(a);
    tile(5, 5) = 100;
}

int main(int argc, char** argv) {
    int a[36] = {0};
    int *d_a;

    cudaMalloc(&d_a, sizeof(int) * 36);
    cudaMemcpy(d_a, a, sizeof(int) * 36, cudaMemcpyHostToDevice);

    print<<<1, 1>>>(d_a);
    cudaDeviceSynchronize();

    cudaMemcpy(a, d_a, sizeof(int) * 36, cudaMemcpyDeviceToHost);

    using Layout6x6 = ct::Layout<ct::Shape<6, 6>>;

    {
        ct::Tile<int, Layout6x6> tile(a);
        for (int i = 0; i < tile.size<0>(); i++) {
            for (int j = 0; j < tile.size<1>(); j++) {
                tile(i, j) = i * 6 + j;
                std::cout << tile(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    using Layout3x3 = ct::Layout<ct::Shape<3, 3>, ct::Stride<9, 3>, Layout6x6>;
    {
        ct::Tile<int, Layout3x3> tile(a);
        for (int i = 0; i < tile.size<0>(); i++) {
            for (int j = 0; j < tile.size<1>(); j++) {
                std::cout << tile(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    using Layout1x3 = ct::Layout<ct::Shape<1, 3>, ct::Shape<1, 3>::stride_t, Layout3x3>;
    {
        ct::Tile<int, Layout1x3> tile(a);
        tile.advance();
        for (int i = 0; i < tile.size<0>(); i++) {
            for (int j = 0; j < tile.size<1>(); j++) {
                std::cout << tile(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }


    return 0;
}