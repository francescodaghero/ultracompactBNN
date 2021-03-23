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
    const uint32_t *pInBuffer;
    const uint32_t *pWeights = weights;
    uint32_t *pOut;
    const int16_t  *thr  = thresholds;
    const uint32_t *golden = checksum;

    BEGIN_STATS_LOOP();
    pInBuffer  = input_data_int1;
    pWeights = weights;
    pOut = output;
    START_STATS();
    conv_bin_1D_nopad_nodilation(
            pInBuffer,
            L_IN,
            C_IN,
            pWeights,
            C_OUT,
            K_S,
            STRIDE,
            pOut,
            L_OUT,
            thr
    );
    STOP_STATS();
    END_STATS_LOOP();
    printf("Errors:\n");
    for (int i = 0; i < (L_OUT * C_OUT + 31) / 32; i++) {
        if (output[i] != checksum[i]) {
            printf("%d out of %d] %x VS %x\n", i, ((L_OUT * C_OUT + 31) / 32) - 1, output[i], checksum[i]);
        }
    }
    //printf("%f \n%f\n",(float)C_IN*L_OUT*C_OUT*K_S, (float)(_cycles/REPEAT));
    printf("MAC/Cycles = %.04f\n", (float)(C_IN*L_OUT*C_OUT*K_S)/(_cycles/REPEAT));
}
#endif
#ifdef CODESIZE
void layer_test() {
    conv_bin_1D_nopad_nodilation(
            input_data_int1,
            L_IN,
            C_IN,
            weights,
            C_OUT,
            K_S,
            STRIDE,
            output,
            L_OUT,
            thresholds
    );
}
#endif





int main() {
   // rt_event_sched_t sched;
   // rt_event_sched_init(&sched);
   // if (rt_event_alloc(&sched, 8)) return -1;
    layer_test();
    //test_stall();
    return 0;
}
