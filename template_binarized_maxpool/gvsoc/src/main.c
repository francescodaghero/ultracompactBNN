#include "rt/rt_api.h"
//#define CODESIZE
#include "../include/data.h"
#include "../include/kernels.h"
#include "../include/data_allocation.h"

#ifndef CODESIZE
#include "../include/stats.h"
void layer_test()
{


    INIT_STATS();

    uint32_t *pInBuffer;
    int32_t *pOut;
    uint32_t *golden = checksum;


    BEGIN_STATS_LOOP();
    pInBuffer  = input_data_int1;
    pOut = output;
    START_STATS();

    maxpool_1d_1bit_w2_fullstrided(
            pInBuffer,
            L_IN,
            C_IN,
            K_S,
            STRIDE,
            pOut,
    L_OUT);

    STOP_STATS();


    END_STATS_LOOP();
    printf("Errors:\n");
    for(int i=0;i < (L_OUT * C_IN + 31) /32; i++) {
        if(output[i]!=checksum[i]) {
            printf("%d out of %d] %x VS %x\n", i,( (L_OUT * C_IN + 31) /32) -1, output[i], checksum[i]);

        }
    }
    printf("MAC/Cycles = %.04f\n", (float)(L_OUT*C_IN*K_S)/(_cycles/REPEAT));


}
#endif

#ifdef CODESIZE
void layer_test()
{
    maxpool_1d_1bit_w2_fullstrided(
            input_data_int1,
            L_IN,
            C_IN,
            K_S,
            STRIDE,
            output,
    L_OUT);


}
#endif

int main() {
    layer_test();

    return 0;
}
