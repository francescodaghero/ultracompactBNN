#ifndef BIN_CONVOLUTION_KERNELS_H
#define BIN_CONVOLUTION_KERNELS_H
void conv_8x1bits_1D_nopad_nodilation_pooling(
        const int8_t * pInBuffer,
        const uint16_t  dim_in,
        const uint16_t  ch_in,
        const int8_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  kernel_size,
        const uint16_t  stride,
        uint32_t *       pOutBuffer,
        const uint16_t  dim_out,
        const int16_t *thresholds,
        const uint8_t pooling_l_out,
        const uint8_t pooling_window,
        const uint8_t pooling_stride
);

void matmul4x2_pooling(
        const int8_t * pInBuffer,
        const uint16_t  dim_in,
        const uint16_t  ch_in,
        const int8_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  kernel_size,
        const uint16_t  stride,
        uint32_t *       pOut1,
        uint32_t output_offset1,
        const uint16_t  dim_out,
        const int16_t *thresholds,
        uint8_t pooling_elements_left,
        uint8_t pooling_window,
        uint8_t pooling_stride
);

void matmul4x1_pooling(
        const int8_t * pInBuffer,
        const uint16_t  dim_in,
        const uint16_t  ch_in,
        const int8_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  kernel_size,
        const uint16_t  stride,
        uint32_t *       pOut1,
        uint32_t output_offset1,
        const uint16_t  dim_out,
        const int16_t *thresholds,
        uint8_t pooling_elements_left,
        uint8_t pooling_window,
        uint8_t pooling_stride
);


#endif //BIN_CONVOLUTION_KERNELS_H
