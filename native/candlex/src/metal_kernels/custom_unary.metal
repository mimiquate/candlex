#include <metal_math>

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

#define CUSTOM_UNARY(IN_TYPE, OUT_TYPE, FN_NAME, FN) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const IN_TYPE *input,  \
    device OUT_TYPE *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = OUT_TYPE(FN(IN_TYPE(input[tid]))); \
}\
kernel void FN_NAME##_strided( \
    constant size_t &dim, \
    constant size_t &num_dims, \
    constant size_t *dims, \
    constant size_t *strides, \
    device const IN_TYPE *input,  \
    device OUT_TYPE *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = OUT_TYPE(FN(IN_TYPE(input[get_strided_index(tid, num_dims, dims, strides)]))); \
}

#define CUSTOM_UNARY_OP(FN_NAME, FN) \
CUSTOM_UNARY(float, float, FN_NAME##_f32, FN);\
CUSTOM_UNARY(half, half, FN_NAME##_f16, FN);

#define CUSTOM_UNARY_BOOL_OP(FN_NAME, FN) \
CUSTOM_UNARY(float, uint8_t, FN_NAME##_f32, FN);\
CUSTOM_UNARY(half, uint8_t, FN_NAME##_f16, FN);

CUSTOM_UNARY_OP(acos, acos)
CUSTOM_UNARY_OP(acosh, acosh)
CUSTOM_UNARY_OP(asin, asin)
CUSTOM_UNARY_OP(asinh, asinh)
CUSTOM_UNARY_OP(atan, atan)
CUSTOM_UNARY_OP(atanh, atanh)
CUSTOM_UNARY_OP(cosh, cosh)
CUSTOM_UNARY_OP(sign, sign)
CUSTOM_UNARY_OP(sinh, sinh)
CUSTOM_UNARY_OP(tan, tan)

CUSTOM_UNARY_BOOL_OP(is_inf, isinf)
CUSTOM_UNARY_BOOL_OP(is_nan, isnan)

CUSTOM_UNARY(uint8_t, uint8_t, bit_not_u8, not)

#if __METAL_VERSION__ >= 220
CUSTOM_UNARY(int64_t, int64_t, sign_i64, sign)
CUSTOM_UNARY(int64_t, int64_t, bit_not_i64, not)
#endif

/* bit_not */
/* cbrt */
/* erfc */
/* erf_inv */
/* expm1 */
/* ln_1p */
/* sigmoid */
