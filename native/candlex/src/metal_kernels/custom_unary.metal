#include <metal_stdlib>
#include <metal_math>

using namespace metal;

#define CUSTOM_UNARY(TYPENAME, FN_NAME, FN) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const TYPENAME *input,  \
    device TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = TYPENAME(FN(float(input[tid]))); \
}

#define CUSTOM_UNARY_OP(NAME) \
CUSTOM_UNARY(float, NAME##_f32, NAME);

#define CUSTOM_UNARY_OP_OUT(TYPENAME, OUT_TYPENAME, FN_NAME, FN) \
kernel void FN_NAME( \
    constant size_t &dim, \
    device const TYPENAME *input,  \
    device OUT_TYPENAME *output, \
    uint tid [[ thread_position_in_grid ]] \
) { \
    if (tid >= dim) { \
        return; \
    } \
    output[tid] = OUT_TYPENAME(FN(float(input[tid]))); \
}

CUSTOM_UNARY_OP(acos)
CUSTOM_UNARY_OP(acosh)
CUSTOM_UNARY_OP(asin)
CUSTOM_UNARY_OP(asinh)
CUSTOM_UNARY_OP(atan)
CUSTOM_UNARY_OP(atanh)
CUSTOM_UNARY_OP(cosh)
CUSTOM_UNARY_OP(sign)
CUSTOM_UNARY_OP(sinh)
CUSTOM_UNARY_OP(tan)

/* bit_not */
/* cbrt */
/* erfc */
/* erf_inv */
/* expm1 */
/* ln_1p */
/* sigmoid */

CUSTOM_UNARY_OP_OUT(float, uint8_t, is_inf_f32, isinf)
CUSTOM_UNARY_OP_OUT(float, uint8_t, is_nan_f32, isnan)
