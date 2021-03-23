#ifndef BIN_CONVOLUTION_KERNELS_H
#define BIN_CONVOLUTION_KERNELS_H

#ifdef TEST_LAYERS
void maxpool_1d_1bit_w2(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint16_t  ch_in,
        const uint16_t  kernel_size,
        const uint16_t  stride,
        uint32_t *       pOutBuffer,
        const uint16_t  dim_out
);

void maxpool_1d_1bit_w2_strided(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint16_t  ch_in,
        const uint16_t  kernel_size,
        const uint16_t  stride,
        uint32_t *       pOutBuffer,
        const uint16_t  dim_out
);
#endif

inline void maxpool_1d_1bit_w2_fullstrided(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint16_t  ch_in,
        const uint16_t  kernel_size,
        const uint16_t  stride,
        uint32_t *       pOutBuffer,
        const uint16_t  dim_out
);

#endif //BIN_CONVOLUTION_KERNELS_H
