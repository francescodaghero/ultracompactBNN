#ifndef BIN_CONVOLUTION_KERNELS_H
#define BIN_CONVOLUTION_KERNELS_H
#include "stdint.h"
void linear(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint32_t *  pWeight,
        const uint16_t  dim_out,
        int32_t *       pOutBuffer,
        const int32_t *thresholds
);
void xnorpop4x1_linear(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint32_t *  pWeight,
        const uint16_t  dim_out,
        int32_t *       pOutBuffer,
        const int32_t *thresholds
);
#ifdef TEST_CORES
void xnorpop3x1(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint32_t *  pWeight,
        const uint16_t  dim_out,
        int32_t *       pOutBuffer,
        const int32_t *thresholds
);

void xnorpop2x1(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint32_t *  pWeight,
        const uint16_t  dim_out,
        int32_t *       pOutBuffer,
        const int32_t *thresholds
);

void xnorpop1x1(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint32_t *  pWeight,
        const uint16_t  dim_out,
        int32_t *       pOutBuffer,
        const int32_t *thresholds
);
#endif
#endif //BIN_CONVOLUTION_KERNELS_H
