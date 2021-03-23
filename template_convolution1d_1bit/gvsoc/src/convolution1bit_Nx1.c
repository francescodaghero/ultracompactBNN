#include "rt/rt_api.h"
#include "../include/kernels.h"
//#include "stdio.h"
//#include <stdint.h>

void conv_bin_1D_nopad_nodilation(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint16_t  ch_in,
        const uint32_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  kernel_size,
        const uint16_t  stride,
        uint32_t *       pOutBuffer,
        const uint16_t  dim_out,
        const int16_t *thresholds
)
{

    uint32_t input_offset = 0;
    uint32_t output_offset = 0;
    uint32_t *pOut = pOutBuffer;

    for(int ts_out = 0; ts_out < dim_out; ts_out++) {

        pOut = xnorpop4x1(
            pInBuffer,
            input_offset,
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

        input_offset += ch_in;
        pInBuffer += input_offset/32;
        input_offset %=32;

        output_offset += ch_out;
        output_offset %=32;
    }




}