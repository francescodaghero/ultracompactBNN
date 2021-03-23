#include "rt/rt_api.h"
#include "../include/kernels.h"
#include "stdio.h"
#include <stdint.h>

//MACROS for pulp
//#define POPCOUNT(x) __builtin_popcount(x)
//#define ROTR(x,bits) rotate_right(x,bits)
#define POPCOUNT(x) __builtin_pulp_cnt(x)
#define ROTR(x,bits) __builtin_pulp_rotr(x,bits)

//MACROS for fast division and modulus
#define DIVIDE_32(x) x>>5
#define MODULO_32(x) (x & 0x1F)
#define XNOR(B,A) (~(B ^ A))

#define DEBUG 0


static inline uint32_t rotate_right (uint32_t u, size_t r)
{
    __asm__ ("rorl %%cl, %0" : "+r" (u) : "c" (r));
    return u;
}

uint32_t *xnorpop1x1(
        const uint32_t * pInBuffer,
        const uint32_t input_offset,
        const uint16_t  dim_in,
        const uint16_t  ch_in,
        const uint32_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  kernel_size,
        const uint16_t  stride,
        uint32_t *       pOut,
        uint32_t output_offset,
        const uint16_t  dim_out,
        const int16_t *thresholds
) {

    int dim_ker=ch_in*kernel_size;

    uint32_t next_pA = dim_ker>=32;
    uint32_t words_input_kernel = DIVIDE_32(dim_ker);
    uint32_t words_leftover_kernel = MODULO_32(ch_in*kernel_size);
    uint32_t leftover_mask = words_leftover_kernel!=0 ? ~((1 << (32 - words_leftover_kernel )) - 1) : 0x0;

    //Input variables
    uint32_t *pB1 = pInBuffer;
    uint32_t *pB1_next;
    uint32_t B1, B1_next;

    //Weights variables
    uint32_t *pA1;
    uint32_t *pA1_next;
    uint32_t A1, A1_next;
    int accum1;


    pA1 = pWeight;
    A1 = (*pA1);
    pA1_next = pA1 + (next_pA);
    A1_next = *(pA1_next);

    uint32_t weights_overstep =0;
    uint32_t weights_mask = (1<< (weights_overstep)) - 1;


    //Variabile per l'output
    uint32_t z = 0;
    uint32_t savecounter = output_offset;
    uint32_t out_mask = 0x80000000;

    //Iterazioni sui canali in output
    for(int out_channel=0; out_channel<ch_out; out_channel++) {

        if(DEBUG) {
            printf("-------------------------\n");
        }
        //Input restarts always from the same timestep
        B1 = *(pB1);
        pB1_next = pB1 + 1;
        B1_next = *(pB1_next);
        B1 <<= input_offset;
        uint32_t input_mask = (1<< input_offset) - 1;
        B1= B1| (B1_next >> (32  - input_offset) & input_mask);


        //Store the conv output
        accum1 =0;



        for (int words_in = 0; words_in < words_input_kernel; words_in++) {

            accum1 += POPCOUNT(XNOR(B1,A1));

            if(DEBUG) {
                printf("A:%x\n", A1);
                printf("B:%x\n",B1);
            }

            //Next words are already loaded
            B1 = B1_next;
            B1 <<= input_offset;
            B1_next = *(++pB1_next);
            B1= (B1) | ( (B1_next >> (32  - input_offset)) & input_mask);


            A1 = A1_next;
            A1 <<= weights_overstep;
            //A2 deve sempre puntare alla parola con i prossimi pesi necessari al seguente calcolo
            uint32_t incr = ((words_leftover_kernel + weights_overstep) >= 32) | ( words_in != (words_input_kernel - 1));
            pA1_next += incr;
            A1_next = (*(pA1_next));
            A1 = A1 | ( (A1_next >> (32-weights_overstep)) & weights_mask);

        }


        //ODD KERNELS or DIM_KER < 32
//        uint32_t mul = ;
        if(DEBUG) {
            printf("A:%x\n", A1);
            printf("B:%x\n",B1);

        }
        accum1 += POPCOUNT(XNOR(B1,A1) & leftover_mask);

        if(DEBUG) {
            printf("ACCUM:%d\n", accum1);
        }

        //THRESHOLDING AND BINARIZATION
        if(accum1 >= *(thresholds++)) {
            z |= out_mask;
        }


        weights_overstep = MODULO_32(weights_overstep + dim_ker);

        //Next set of weights
        A1 = A1_next;
        A1 <<= weights_overstep;
        pA1_next += ((weights_overstep + dim_ker) >=32) | (next_pA);
        A1_next = *(pA1_next);
        weights_mask = (1<< (weights_overstep)) - 1;
        A1 = A1 |  (A1_next >> (32-weights_overstep) & weights_mask);


        //TODO Find a smarter way to do this
        savecounter +=1;
        out_mask = ROTR(out_mask,1);
        if((savecounter)%32==0) {
            *(pOut++)|=(z>>output_offset);
            savecounter=0;
            z = 0;
        }

    }

    if(z!=0)
    {
        *(pOut) |= (z>>output_offset);
        pOut += ((output_offset+ch_out)==32);

    }


    return pOut;


}