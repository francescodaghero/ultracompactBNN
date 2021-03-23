#ifndef PULP_NN_DATA_ALLOCATION_H
#define PULP_NN_DATA_ALLOCATION_H
RT_FC_SHARED_DATA const uint32_t input_data_int1[(L_IN + 31)>>5]  = INPUT;
RT_FC_SHARED_DATA const uint32_t weights[(L_OUT*L_IN + 31)>>5] = WEIGHTS;
#ifndef CODESIZE
const uint32_t checksum[L_OUT]  = OUTPUT;
#endif
static int32_t output[L_OUT]  = {0};
//const int32_t checksum_fp[L_OUT]  = OUTPUT_FP;
const int32_t thresholds[(L_OUT<<1)] = THRESHOLDS;
#endif //PULP_NN_DATA_ALLOCATION_H