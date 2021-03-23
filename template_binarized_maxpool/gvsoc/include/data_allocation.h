#ifndef PULP_NN_DATA_ALLOCATION_H
#define PULP_NN_DATA_ALLOCATION_H

const uint32_t input_data_int1[(C_IN * L_IN + 31)/32]  = INPUT;
#ifndef CODESIZE
const uint32_t checksum[(C_IN * L_OUT + 31)/32]  = OUTPUT;
#endif
static uint32_t output[(C_IN * L_OUT+ 31)/32]  = {0};

#endif //PULP_NN_DATA_ALLOCATION_H