#ifndef BIN_CONVOLUTION_KERNELS_H
#define BIN_CONVOLUTION_KERNELS_H
void conv_bin_1D_nopad_nodilation_pooling(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint16_t  ch_in,
        const uint32_t *  pWeight,
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

#ifdef TEST_KERNELS
uint32_t *xnorpop1x4(
        const uint32_t * pInBuffer,
        const uint32_t input_offset1,
        const uint16_t  dim_ker,//
        const uint16_t  ch_in,
        const uint32_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  stride,
        uint32_t *       pOut1,
        uint32_t output_offset1,
        const int16_t *thresholds
);

uint32_t *xnorpop2x4(
        const uint32_t * pInBuffer,
        const uint32_t input_offset1,
        const uint16_t  dim_ker,//
        const uint16_t  ch_in,
        const uint32_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  stride,
        uint32_t *       pOut1,
        uint32_t output_offset1,
        const int16_t *thresholds
);

uint32_t *xnorpop4x4(
        const uint32_t * pInBuffer,
        const uint32_t input_offset1,
        const int  dim_ker,//
        const uint16_t  ch_in,
        const uint32_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  stride,
        uint32_t *       pOut1,
        uint32_t output_offset1,
        const int16_t *thresholds
);

uint32_t *xnorpop1x2(
        const uint32_t * pInBuffer,
        const uint32_t input_offset1,
        const uint16_t  dim_ker,//
        const uint16_t  ch_in,
        const uint32_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  stride,
        uint32_t *       pOut1,
        uint32_t output_offset1,
        const int16_t *thresholds
);

uint32_t *xnorpop4x2(
        const uint32_t * pInBuffer,
        const uint32_t input_offset1,
        const uint16_t  dim_ker,//
        const uint16_t  ch_in,
        const uint32_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  stride,
        uint32_t *       pOut1,
        uint32_t output_offset1,
        const int16_t *thresholds
);


uint32_t *xnorpop1x1(
        const uint32_t * pInBuffer,
        const uint32_t input_offset1,
        const uint16_t  dim_ker,//
        const uint16_t  ch_in,
        const uint32_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  stride,
        uint32_t *       pOut1,
        uint32_t output_offset1,
        const int16_t *thresholds
);

uint32_t *xnorpop2x1(
        const uint32_t * pInBuffer,
        const uint32_t input_offset1,
        const uint16_t  dim_ker,//
        const uint16_t  ch_in,
        const uint32_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  stride,
        uint32_t *       pOut1,
        uint32_t output_offset1,
        const int16_t *thresholds
);
#endif
void xnorpop2x2_pooling(
        const uint32_t * pInBuffer,
        const uint32_t input_offset1,
        const uint16_t  dim_ker,//
        const uint16_t  ch_in,
        const uint32_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  stride,
        uint32_t *       pOut1,
        uint32_t output_offset1,
        const int16_t *thresholds,
        uint8_t pooling_elements_left,
        uint8_t pooling_window,
        uint8_t pooling_stride
);

void xnorpop4x1_pooling(
        const uint32_t * pInBuffer,
        const uint32_t input_offset1,
        const uint16_t  dim_ker,//
        const uint16_t  ch_in,
        const uint32_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  stride,
        uint32_t *       pOut1,
        uint32_t output_offset1,
        const int16_t *thresholds,
        uint8_t pooling_elements_left,
        uint8_t pooling_window,
        uint8_t pooling_stride
);

#endif //BIN_CONVOLUTION_KERNELS_H
