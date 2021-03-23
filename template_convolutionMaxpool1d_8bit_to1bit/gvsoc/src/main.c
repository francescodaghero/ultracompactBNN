#include "rt/rt_api.h"
#include "../include/data.h"
#include "../include/kernels.h"
#include "../include/data_allocation.h"

#ifndef CODESIZE
#include "../include/stats.h"
void layer_test()
{
    INIT_STATS();


    int8_t *pInBuffer;
    int8_t *pWeights = weights;
    uint32_t *pOut;
    int16_t  *thr  = thresholds;
    uint32_t *golden = checksum;


    BEGIN_STATS_LOOP();
    pInBuffer  = input_data_int1;

    pOut = output;
    START_STATS();
    conv_8x1bits_1D_nopad_nodilation_pooling(
            pInBuffer,
            L_IN,
            C_IN,
            pWeights,
            C_OUT,
            K_S,
            STRIDE,
            pOut,
            L_OUT,
            thr,
            POOL_L_OUT,
            POOLING_WINDOW,
            POOLING_STRIDE
    );

    STOP_STATS();



    END_STATS_LOOP();
    printf("Convolution Test Ended\n");
    printf("Errors:\n");
    for (int i = 0; i < (POOL_L_OUT * C_OUT + 31) / 32; i++) {
        if (output[i] != checksum[i]) {
            printf("%d out of %d] %x VS %x\n", i, ((POOL_L_OUT * C_OUT + 31) / 32) - 1, output[i], checksum[i]);

        }
    }
    printf("MAC/Cycles = %.04f\n", (float)(L_OUT*C_OUT*C_IN*K_S)/(_cycles/REPEAT));


}
#endif

#ifdef CODESIZE
void layer_test()
{
    conv_8x1bits_1D_nopad_nodilation_pooling(
                input_data_int1,
                L_IN,
                C_IN,
                weights,
                C_OUT,
                K_S,
                STRIDE,
                output,
                L_OUT,
                thresholds,
                POOL_L_OUT,
                POOLING_WINDOW,
                POOLING_STRIDE
        );
}

#endif


int main() {
    layer_test();


    return 0;
}
