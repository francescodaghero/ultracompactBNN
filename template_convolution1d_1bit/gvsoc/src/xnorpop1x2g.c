#include "rt/rt_api.h"
#include "../include/kernels.h"

//MACROS
#define DIVIDE_32(x) x>>5
#define MODULO_32(x) (x & 0x1F)
#define XNOR(B,A) (~(B ^ A))

#define POPCOUNT(x) __builtin_pulp_cnt(x)

//#define DEBUG


uint32_t *xnorpop1x2(
        const uint32_t * pInBuffer,
        const uint32_t input_offset1,
        const uint16_t  dim_ker,//
        const uint16_t  ch_in,
        const uint32_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  stride,
        uint32_t *       pOut1,
        uint32_t output_offset1,
        const int16_t *thresholds
) {
    uint32_t *pB1_next, B1;
//DECLARE VARIABLES
    uint32_t input_offset2;
    uint32_t *pA1, *pA1_next;
    uint32_t A1;

    uint32_t *pOut2;

    //Variables
    //NxM dependant
    //N
    uint32_t leftover_channels = ch_out %1;
    //M
    input_offset2 = MODULO_32(input_offset1 + ch_in);
    //Other variables
    uint32_t words_kernel = DIVIDE_32(dim_ker);
    uint32_t words_leftover_kernel = MODULO_32(dim_ker);
    uint32_t leftover_mask = words_leftover_kernel!=0 ? ~((1 << (32 - words_leftover_kernel )) - 1) : 0x0;

    uint32_t weights_overstep1 =0;
    uint32_t weights_mask1 = (1<< (weights_overstep1)) - 1;


    uint32_t output_offset2 = MODULO_32(output_offset1 + ch_out);

    pA1 = pWeight;
    pA1_next = pWeight + (dim_ker >= 32);
    pOut2 = pOut1 + (DIVIDE_32((output_offset1 + ch_out)));

    //Load
    A1 = *(pA1);

    if((ch_out)) {
        int out_channel = 0;
        do {
            #ifdef DEBUG
        printf("-------------------------\n");
        #endif
        uint32_t B2;
        //Update pointers and variables
        uint32_t *pB1 = pInBuffer;
        uint32_t *pB2 = pB1  + (DIVIDE_32((input_offset1 + ch_in)));

        uint32_t input_mask1 = (1<< input_offset1) - 1;
        uint32_t input_mask2 = (1<< input_offset2) - 1;

        //Additional weights







        //Store the conv output
        int accum1_ts1 =0;
        int accum1_ts2 =0;

        //LOAD


        //Operations on loaded values
        B1= ((*(pB1++)) << input_offset1)| (*(pB1) >> (32  - input_offset1) & input_mask1);
        B2= ((*(pB2++)) << input_offset2)| (*(pB2) >> (32  - input_offset2) & input_mask2);



        if((words_kernel)>0) {
            for (int words_in = 0; words_in < (words_kernel-1); words_in++) {

                accum1_ts1 += POPCOUNT(XNOR(B1,A1));
                accum1_ts2 += POPCOUNT(XNOR(B2,A1));
                //asm volatile("":::"memory");

                #ifdef DEBUG
                printf("A1:%x\n",A1);
                printf("B1:%x\n",B1);
                printf("B2:%x\n",B2);
                #endif

                //Swap values
                A1 = *(pA1_next++);


                B1= ((*(pB1++)) << input_offset1) | ( (*(pB1) >> (32  - input_offset1)) & input_mask1);
                B2= ((*(pB2++)) << input_offset2) | ( (*(pB2) >> (32  - input_offset2)) & input_mask2);
                A1 <<= weights_overstep1;


                A1 = A1 | ( (*pA1_next >> (32-weights_overstep1 )) & weights_mask1);

            }
            accum1_ts1 += POPCOUNT(XNOR(B1,A1));
            accum1_ts2 += POPCOUNT(XNOR(B2,A1));
            //asm volatile("":::"memory");

            #ifdef DEBUG
                            printf("A1:%x\n",A1);
                            printf("B1:%x\n",B1);
                            printf("B2:%x\n",B2);
            #endif

            //Swap values
            A1 = *(pA1_next);


            B1= ((*(pB1++)) << input_offset1) | ( (*(pB1) >> (32  - input_offset1)) & input_mask1);
            B2= ((*(pB2++)) << input_offset2) | ( (*(pB2) >> (32  - input_offset2)) & input_mask2);
            A1 <<= weights_overstep1;

            pA1_next += ((weights_overstep1 + words_leftover_kernel) >= 32);


            A1 = A1 | ( (*pA1_next >> (32-weights_overstep1 )) & weights_mask1);

        }

        //ODD KERNELS or DIM_KER < 32
        #ifdef DEBUG
        printf("A1:%x\n", A1 & leftover_mask);
        printf("B1:%x\n",B1 & leftover_mask);
        printf("B2:%x\n",B2 & leftover_mask);
        #endif
        accum1_ts1 += POPCOUNT(XNOR(B1,A1) & leftover_mask);
        accum1_ts2 += POPCOUNT(XNOR(B2,A1) & leftover_mask);



        #ifdef DEBUG
        printf("ACCUM1 TS 1:%d\n", accum1_ts1);
        printf("ACCUM1 TS 2:%d\n", accum1_ts2);

        #endif

        //THRESHOLDING AND BINARIZATION
        *pOut1 |= (accum1_ts1 >= *(thresholds)) << (32 - output_offset1 - 1);
        *pOut2 |= (accum1_ts2 >= *(thresholds++)) << (32 - output_offset2 -1);

//      asm volatile("":::"memory");

        weights_overstep1  = MODULO_32(weights_overstep1   + dim_ker);

        A1 = *(pA1_next);
        pA1_next =pA1_next + ( ((weights_overstep1 + words_leftover_kernel) >= 32) | dim_ker >= 32);
        weights_mask1 = (1<< (weights_overstep1)) - 1;
        A1 = (A1 << weights_overstep1) |  (*(pA1_next) >> (32-weights_overstep1) & weights_mask1);


        output_offset1+= 1;
        pOut1 += DIVIDE_32(output_offset1);
        output_offset1 =MODULO_32(output_offset1);
        output_offset2+= 1;
        pOut2 += DIVIDE_32(output_offset2);
        output_offset2 =MODULO_32(output_offset2);


            out_channel++;
        } while (out_channel < (ch_out));


    }



    return pOut2;


}