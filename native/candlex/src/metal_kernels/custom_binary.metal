using namespace metal;

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
}

CUSTOM_BINARY(int64_t, int64_t, bit_and_i64, x & y)
CUSTOM_BINARY(int64_t, int64_t, bit_or_i64, x | y)
CUSTOM_BINARY(int64_t, int64_t, bit_xor_i64, x ^ y)

CUSTOM_BINARY(float, float, atan2_f32, atan2(x, y))

/* pow */
/* remainder */
/* shl */
/* shr */

CUSTOM_BINARY(int64_t, uint8_t, logical_and_i64, x && y)
CUSTOM_BINARY(uint8_t, uint8_t, logical_and_u8, x && y)
CUSTOM_BINARY(int64_t, uint8_t, logical_or_i64, x || y)
CUSTOM_BINARY(uint8_t, uint8_t, logical_or_u8, x || y)
CUSTOM_BINARY(int64_t, uint8_t, logical_xor_i64, !x != !y)
CUSTOM_BINARY(uint8_t, uint8_t, logical_xor_u8, !x != !y)
