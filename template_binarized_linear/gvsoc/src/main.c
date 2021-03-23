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
    uint32_t *pWeights = weights;
    int32_t *pOut;
    int32_t  *thr  = thresholds;
    uint32_t *golden = checksum;


    BEGIN_STATS_LOOP();
    pInBuffer  = input_data_int1;
    pWeights = weights;
    pOut = output;
    START_STATS();
    linear(
            pInBuffer,
            L_IN,
            pWeights,
            L_OUT,
            pOut,
            thresholds
    );

    STOP_STATS();


    END_STATS_LOOP();
    printf("Errors:\n");
    for (int i = 0; i < L_OUT; i++) {
        if(checksum[i]!=output[i]) {
            printf("Output: %d\n", output[i]);
        }
    }
    printf("MAC/Cycles = %.04f\n", (float)(L_IN*L_OUT)/(_cycles/REPEAT));


}
#endif

#ifdef CODESIZE
void layer_test()
{
   linear(
            input_data_int1,
            L_IN,
            weights,
            L_OUT,
            output,
            thresholds
    );
}
#endif

int main() {
//    rt_event_sched_t sched;
//    rt_event_sched_init(&sched);
//    if (rt_event_alloc(&sched, 8)) return -1;
    layer_test();
    return 0;
}



