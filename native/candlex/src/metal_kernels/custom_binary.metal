using namespace metal;

METAL_FUNC uint get_strided_index(
    uint idx,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides
) {
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

#define CUSTOM_BINARY(IN_TYPE, OUT_TYPE, FN_NAME, FN) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const IN_TYPE *left,  \
    device const IN_TYPE *right,  \
    device OUT_TYPE *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    IN_TYPE x = left[tid]; \
    IN_TYPE y = right[tid]; \
    output[tid] = OUT_TYPE(FN); \
}\
kernel void FN_NAME##_strided( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *left_strides, \
    constant size_t *right_strides, \
    device const IN_TYPE *left,  \
    device const IN_TYPE *right,  \
    device OUT_TYPE *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    IN_TYPE x = left[get_strided_index(tid, num_dims, dims, left_strides)]; \
    IN_TYPE y = right[get_strided_index(tid, num_dims, dims, right_strides)]; \
    output[tid] = OUT_TYPE(FN); \
}

#define CUSTOM_BINARY_OP(FN_NAME, FN)\
CUSTOM_BINARY(float, float, FN_NAME##_f32, FN);\
CUSTOM_BINARY(half, half, FN_NAME##_f16, FN);\
CUSTOM_BINARY(uint32_t, uint32_t, FN_NAME##_u32, FN);\
CUSTOM_BINARY(uint8_t, uint8_t, FN_NAME##_u8, FN);

#define CUSTOM_BINARY_BOOL_OP(FN_NAME, FN)\
CUSTOM_BINARY(float, uint8_t, FN_NAME##_f32, FN);\
CUSTOM_BINARY(half, uint8_t, FN_NAME##_f16, FN);\
CUSTOM_BINARY(uint32_t, uint8_t, FN_NAME##_u32, FN);\
CUSTOM_BINARY(uint8_t, uint8_t, FN_NAME##_u8, FN);

CUSTOM_BINARY_OP(atan2, atan2(x, y))
CUSTOM_BINARY_OP(pow, pow(x, y))
CUSTOM_BINARY_OP(bit_and, x & y)
CUSTOM_BINARY_OP(bit_or, x | y)
CUSTOM_BINARY_OP(bit_xor, x ^ y)
CUSTOM_BINARY_OP(shl, x << y)
CUSTOM_BINARY_OP(shr, x >> r)

CUSTOM_BINARY_BOOL_OP(logical_and, x && y)
CUSTOM_BINARY_BOOL_OP(logical_or, x || y)
CUSTOM_BINARY_BOOL_OP(logical_xor, !x != !y)

#if __METAL_VERSION__ >= 220
CUSTOM_BINARY(int64_t, int64_t, bit_and_i64, x & y)
CUSTOM_BINARY(int64_t, int64_t, bit_or_i64, x | y)
CUSTOM_BINARY(int64_t, int64_t, bit_xor_i64, x ^ y)
CUSTOM_BINARY(int64_t, int64_t, shl_i64, x << y)
CUSTOM_BINARY(int64_t, int64_t, shr_i64, x >> y)

CUSTOM_BINARY(int64_t, uint8_t, logical_and_i64, x && y)
CUSTOM_BINARY(int64_t, uint8_t, logical_or_i64, x || y)
CUSTOM_BINARY(int64_t, uint8_t, logical_xor_i64, !x != !y)
#endif

/* remainder */
