#include <metal_math>

using namespace metal;

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
}

CUSTOM_UNARY(float, float, acos_f32, acos)
CUSTOM_UNARY(float, float, acosh_f32, acosh)
CUSTOM_UNARY(float, float, asin_f32, asin)
CUSTOM_UNARY(float, float, asinh_f32, asinh)
CUSTOM_UNARY(float, float, atan_f32, atan)
CUSTOM_UNARY(float, float, atanh_f32, atanh)
CUSTOM_UNARY(float, float, cosh_f32, cosh)
CUSTOM_UNARY(float, float, sign_f32, sign)
CUSTOM_UNARY(float, float, sinh_f32, sinh)
CUSTOM_UNARY(float, float, tan_f32, tan)

/* bit_not */
/* cbrt */
/* erfc */
/* erf_inv */
/* expm1 */
/* ln_1p */
/* sigmoid */

CUSTOM_UNARY(float, uint8_t, is_inf_f32, isinf)
CUSTOM_UNARY(float, uint8_t, is_nan_f32, isnan)
