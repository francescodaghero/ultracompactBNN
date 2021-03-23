#ifndef PULP_NN_DATA_ALLOCATION_H
#define PULP_NN_DATA_ALLOCATION_H
RT_FC_SHARED_DATA const uint32_t input_data_int1[(C_IN * L_IN + 31)/32]  = INPUT;
RT_FC_SHARED_DATA const uint32_t weights[(C_OUT*K_S*C_IN+31)/32] = WEIGHTS;
static uint32_t output[(C_OUT * L_OUT+ 31)/32]  = {0};
//static int32_t checksum_fp[C_OUT*L_OUT]  = OUTPUT_FP;
const int16_t thresholds[(C_OUT)] = THRESHOLDS;
#ifndef CODESIZE
const uint32_t checksum[(C_OUT * L_OUT + 31)/32]  = OUTPUT;
#endif

#endif //PULP_NN_DATA_ALLOCATION_H