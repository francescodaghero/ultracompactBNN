//#include "stdio.h"
#include "rt/rt_api.h"
#include "../include/kernels.h"

void __attribute__((always_inline)) conv_8x1bits_1D_nopad_nodilation_pooling(
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
)
{

    uint32_t ts_timesteps = 2;
    uint32_t dim_out_leftover = (pooling_l_out*pooling_window) % ts_timesteps;
    int out_timestep = (pooling_l_out*pooling_window)>>1;//dim_out>>1;

    uint32_t output_offset = 0;
    uint32_t *pOut = pOutBuffer;

    int input_increase = (ts_timesteps * ch_in * stride);

    uint8_t pooling_counter=0;
    for(int ts_out = 0; ts_out < out_timestep; ts_out++) {
        matmul4x2_pooling(
                pInBuffer,
                dim_in,
                ch_in,
                pWeight,
                ch_out,
                kernel_size,
                stride,
                pOut,
                output_offset,
                dim_out,
                thresholds,
                pooling_counter,
                pooling_window,
                pooling_stride
        );

        pooling_counter+=ts_timesteps;

        pInBuffer += input_increase;

        output_offset += (ch_out) * (pooling_counter>=pooling_window);
//        printf("%d___%d___%d\n", output_offset, pooling_counter, pooling_window);
        pOut+= output_offset>>5;
        output_offset %=32;

        pooling_counter%=pooling_window;



    }

    while(dim_out_leftover)
    {
        matmul4x1_pooling(
                pInBuffer,
                dim_in,
                ch_in,
                pWeight,
                ch_out,
                kernel_size,
                stride,
                pOut,
                output_offset,
                dim_out,
                thresholds,
                pooling_counter,
                pooling_window,
                pooling_stride
        );

        pooling_counter+=ts_timesteps;
        output_offset += (ch_out) * (pooling_counter>=pooling_window);
        pOut+= output_offset>>5;
        output_offset %=32;
        pooling_counter%=pooling_window;

        dim_out_leftover--;
    }


}
