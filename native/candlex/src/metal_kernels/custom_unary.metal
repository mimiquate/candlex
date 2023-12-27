#include <metal_stdlib>

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

CUSTOM_UNARY_OP_OUT(float, uint8_t, is_inf_f32, isinf(x) ? 1 : 0)
