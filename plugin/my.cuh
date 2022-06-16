#include <cublas_v2.h>
#include <cub/cub.cuh>


template <int VPT>
struct BytesToType;
// 
template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};


template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;
// 
    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}


__device__ inline float2 operator + (const  float2 &a, const float2 & b) {
  float2 out{0.0f, 0.0f};
  printf("a.x %f, b.x %f\t a.y %f, b.y %f\n", a.x, b.x, a.y, b.y);
  out.x = a.x + b.x;
  out.y = a.y + b.y;
  return out;
}


template <typename T>
using kvp = cub::KeyValuePair<T, T>;


using kv_float = cub::KeyValuePair<float, float>;
using kv_half = cub::KeyValuePair<half, half>;
using kv_half2 = cub::KeyValuePair<half2, half2>;


__device__ inline kv_float operator+(const kv_float& a, const kv_float& b)
{
    return kv_float(a.key + b.key, a.value + b.value);
}



template <typename T>
__device__ inline T operator+(const T& a, const T& b);