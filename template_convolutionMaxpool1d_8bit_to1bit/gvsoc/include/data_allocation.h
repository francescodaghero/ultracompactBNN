#ifndef PULP_NN_DATA_ALLOCATION_H
#define PULP_NN_DATA_ALLOCATION_H

int8_t input_data_int1[(C_IN * L_IN)]  = INPUT;
int8_t weights[(C_OUT*K_S*C_IN)] = WEIGHTS;
uint32_t output[(C_OUT * L_OUT+ 31)/32]  = {0};
#ifndef CODESIZE
int32_t checksum_fp[C_OUT*L_OUT]  = OUTPUT_FP;
uint32_t checksum[(C_OUT * L_OUT + 31)/32]  = OUTPUT;
#endif
int16_t thresholds[(C_OUT)] = THRESHOLDS;

#endif //PULP_NN_DATA_ALLOCATION_H