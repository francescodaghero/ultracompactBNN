//#include "stdio.h"
#include "rt/rt_api.h"
#include "../include/kernels.h"

void __attribute__((always_inline)) conv_8x1bits_1D_nopad_nodilation(
        const int8_t * pInBuffer,
        const uint16_t  dim_in,
        const uint16_t  ch_in,
        const int8_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  kernel_size,
        const uint16_t  stride,
        uint32_t *       pOutBuffer,
        const uint16_t  dim_out,
        const int16_t *thresholds
)
{

    uint32_t ts_timesteps = 2;
    uint32_t dim_out_leftover = dim_out % ts_timesteps;

    uint32_t output_offset = 0;
    uint32_t *pOut = pOutBuffer;

    int input_increase = (ts_timesteps * ch_in * stride);

    for(int ts_out = 0; ts_out < dim_out>>1; ts_out++) {
        pOut = matmul4x2(
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
                thresholds
        );

        pInBuffer += input_increase;

        output_offset += (ts_timesteps*ch_out);
        output_offset %=32;


    }

    while(dim_out_leftover)
    {
        pOut = matmul4x1(
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
                thresholds
        );
        dim_out_leftover--;
    }


}
