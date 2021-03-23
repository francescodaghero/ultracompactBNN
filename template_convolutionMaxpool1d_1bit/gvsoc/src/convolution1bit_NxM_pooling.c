#include "rt/rt_api.h"
#include "../include/kernels.h"
//#include "stdio.h"
//#include <stdint.h>
void __attribute__((always_inline)) conv_bin_1D_nopad_nodilation_pooling(
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
)
{
    uint32_t ts_timesteps = 2;
    int out_timestep = (pooling_l_out*pooling_window)>>1;//dim_out>>1;
    uint32_t dim_out_leftover = dim_out % ts_timesteps;

    int input_offset = 0;
    int output_offset = 0;
    uint32_t *pOut = pOutBuffer;
    int shift_input = (ts_timesteps*ch_in);
    int output_offset_shift = (ts_timesteps*ch_out);
    const uint32_t dim_ker = ch_in*kernel_size;

    uint8_t pooling_counter=0;
    for(int ts_out = 0; ts_out < out_timestep; ts_out++) {

        xnorpop2x2_pooling(
            pInBuffer,
            input_offset,
            dim_ker,
            ch_in,
            pWeight,
            ch_out,
            stride,
            pOut,
            output_offset,
            thresholds,
            pooling_counter,
            pooling_window,
            pooling_stride
        );

        pooling_counter+=ts_timesteps;
        input_offset += shift_input;
        pInBuffer += input_offset>>5;
        input_offset %=32;

        output_offset += (ch_out) * (pooling_counter>=pooling_window);
//        printf("%d___%d___%d\n", output_offset, pooling_counter, pooling_window);
        pOut+= output_offset>>5;
        output_offset %=32;

        pooling_counter%=pooling_window;

    }

    while(dim_out_leftover)
    {
        xnorpop4x1_pooling(
                pInBuffer,
                input_offset,
                dim_ker,
                ch_in,
                pWeight,
                ch_out,
                stride,
                pOut,
                output_offset,
                thresholds,
                pooling_counter,
                pooling_window,
                pooling_stride
        );

        pooling_counter+=ts_timesteps;

        input_offset += ch_in;
        pInBuffer += input_offset>>5;
        input_offset %=32;
//        output_offset += ch_out;
        output_offset += (ch_out) * (pooling_counter>=pooling_window);
        pOut+= output_offset>>5;
        output_offset %=32;
        pooling_counter%=pooling_window;


        dim_out_leftover--;

    }
}