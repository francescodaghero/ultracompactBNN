#include "rt/rt_api.h"
#include "../include/kernels.h"

//MACROS
#define DIVIDE_32(x) x>>5
#define MODULO_32(x) (x & 0x1F)
#define XNOR(B,A) (~(B ^ A))

#define POPCOUNT(x) __builtin_pulp_cnt(x)

//#define DEBUG


uint32_t *xnorpop2x1(
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
    uint32_t *pA1, *pA1_next;
    uint32_t A1;


    //Variables
    //NxM dependant
    //N
    uint32_t leftover_channels = ch_out %2;
    //M
    //Other variables
    uint32_t words_kernel = DIVIDE_32(dim_ker);
    uint32_t words_leftover_kernel = MODULO_32(dim_ker);
    uint32_t leftover_mask = words_leftover_kernel!=0 ? ~((1 << (32 - words_leftover_kernel )) - 1) : 0x0;

    uint32_t weights_overstep1 =0;
    uint32_t weights_mask1 = (1<< (weights_overstep1)) - 1;



    pA1 = pWeight;
    pA1_next = pWeight + (dim_ker >= 32);

    //Load
    A1 = *(pA1);

    if(((ch_out)>>1)) {
        int out_channel = 0;
        do {
            #ifdef DEBUG
        printf("-------------------------\n");
        #endif
        //Update pointers and variables
        uint32_t *pB1 = pInBuffer;

        uint32_t input_mask1 = (1<< input_offset1) - 1;

        //Additional weights
        uint32_t weights_overstep2 = MODULO_32(weights_overstep1)+words_leftover_kernel;

        uint32_t *pA2 = pA1 + words_kernel + (weights_overstep2 >=32); //Point pAI all'inizio dei pesi

        weights_overstep2 = MODULO_32(weights_overstep2);



        uint32_t *pA2_next = pA2 + (((weights_overstep2 + words_leftover_kernel) >= 32) | dim_ker >= 32);

        uint32_t weights_mask2 = (1<< (weights_overstep2)) - 1;

        //Store the conv output
        int accum1_ts1 =0;
        int accum2_ts1 =0;

        //LOAD

        uint32_t A2 = *(pA2);

        //Operations on loaded values
        B1= ((*(pB1++)) << input_offset1)| (*(pB1) >> (32  - input_offset1) & input_mask1);


        A2 = A2 << weights_overstep2 |  (*(pA2_next) >> (32-weights_overstep2) & weights_mask2);

        if((words_kernel)>0) {
            for (int words_in = 0; words_in < (words_kernel-1); words_in++) {

                accum1_ts1 += POPCOUNT(XNOR(B1,A1));
                accum2_ts1 += POPCOUNT(XNOR(B1,A2));
                //asm volatile("":::"memory");

                #ifdef DEBUG
                printf("A1:%x\n",A1);
                printf("A2:%x\n",A2);
                printf("B1:%x\n",B1);
                #endif

                //Swap values
                A1 = *(pA1_next++);
                A2 = *(pA2_next++);


                B1= ((*(pB1++)) << input_offset1) | ( (*(pB1) >> (32  - input_offset1)) & input_mask1);
                A1 <<= weights_overstep1;
                A2 <<= weights_overstep2;


                A1 = A1 | ( (*pA1_next >> (32-weights_overstep1 )) & weights_mask1);
                A2 = A2 | ( (*pA2_next >> (32-weights_overstep2 )) & weights_mask2);

            }
            accum1_ts1 += POPCOUNT(XNOR(B1,A1));
            accum2_ts1 += POPCOUNT(XNOR(B1,A2));
            //asm volatile("":::"memory");

            #ifdef DEBUG
                            printf("A1:%x\n",A1);
                            printf("A2:%x\n",A2);
                            printf("B1:%x\n",B1);
            #endif

            //Swap values
            A1 = *(pA1_next);
            A2 = *(pA2_next);


            B1= ((*(pB1++)) << input_offset1) | ( (*(pB1) >> (32  - input_offset1)) & input_mask1);
            A1 <<= weights_overstep1;
            A2 <<= weights_overstep2;

            pA1_next += ((weights_overstep1 + words_leftover_kernel) >= 32);
            pA2_next += ((weights_overstep2 + words_leftover_kernel) >= 32);


            A1 = A1 | ( (*pA1_next >> (32-weights_overstep1 )) & weights_mask1);
            A2 = A2 | ( (*pA2_next >> (32-weights_overstep2 )) & weights_mask2);

        }

        //ODD KERNELS or DIM_KER < 32
        #ifdef DEBUG
        printf("A1:%x\n", A1 & leftover_mask);
        printf("A2:%x\n", A2 & leftover_mask);
        printf("B1:%x\n",B1 & leftover_mask);
        #endif
        accum1_ts1 += POPCOUNT(XNOR(B1,A1) & leftover_mask);
        accum2_ts1 += POPCOUNT(XNOR(B1,A2) & leftover_mask);



        #ifdef DEBUG
        printf("ACCUM1 TS 1:%d\n", accum1_ts1);
        printf("ACCUM2 TS 1:%d\n", accum2_ts1);

        #endif

        //THRESHOLDING AND BINARIZATION
        *pOut1 |= (accum1_ts1 >= *(thresholds++)) << (32 - output_offset1 -1);

        *pOut1 |= (accum2_ts1 >= *(thresholds++)) << (32 - output_offset1 - 2 );
//      asm volatile("":::"memory");

        weights_overstep1  = MODULO_32(weights_overstep2   + dim_ker);

        A1 = *(pA2_next);
        pA1_next =pA2_next + ( ((weights_overstep1 + words_leftover_kernel) >= 32) | dim_ker >= 32);
        weights_mask1 = (1<< (weights_overstep1)) - 1;
        A1 = (A1 << weights_overstep1) |  (*(pA1_next) >> (32-weights_overstep1) & weights_mask1);

        pA1 = pA2_next;

        output_offset1+= 2;
        pOut1 += DIVIDE_32(output_offset1);
        output_offset1 =MODULO_32(output_offset1);


            out_channel++;
        } while (out_channel < (ch_out) >> 1);


    }

    while(leftover_channels) {
    #ifdef DEBUG
        printf("-------------------------\n");
        #endif
        //Update pointers and variables
        uint32_t *pB1 = pInBuffer;

        uint32_t input_mask1 = (1<< input_offset1) - 1;

        //Additional weights







        //Store the conv output
        int accum1_ts1 =0;

        //LOAD


        //Operations on loaded values
        B1= ((*(pB1++)) << input_offset1)| (*(pB1) >> (32  - input_offset1) & input_mask1);



        if((words_kernel)>0) {
            for (int words_in = 0; words_in < (words_kernel-1); words_in++) {

                accum1_ts1 += POPCOUNT(XNOR(B1,A1));
                //asm volatile("":::"memory");

                #ifdef DEBUG
                printf("A1:%x\n",A1);
                printf("B1:%x\n",B1);
                #endif

                //Swap values
                A1 = *(pA1_next++);


                B1= ((*(pB1++)) << input_offset1) | ( (*(pB1) >> (32  - input_offset1)) & input_mask1);
                A1 <<= weights_overstep1;


                A1 = A1 | ( (*pA1_next >> (32-weights_overstep1 )) & weights_mask1);

            }
            accum1_ts1 += POPCOUNT(XNOR(B1,A1));
            //asm volatile("":::"memory");

            #ifdef DEBUG
                            printf("A1:%x\n",A1);
                            printf("B1:%x\n",B1);
            #endif

            //Swap values
            A1 = *(pA1_next);


            B1= ((*(pB1++)) << input_offset1) | ( (*(pB1) >> (32  - input_offset1)) & input_mask1);
            A1 <<= weights_overstep1;

            pA1_next += ((weights_overstep1 + words_leftover_kernel) >= 32);


            A1 = A1 | ( (*pA1_next >> (32-weights_overstep1 )) & weights_mask1);

        }

        //ODD KERNELS or DIM_KER < 32
        #ifdef DEBUG
        printf("A1:%x\n", A1 & leftover_mask);
        printf("B1:%x\n",B1 & leftover_mask);
        #endif
        accum1_ts1 += POPCOUNT(XNOR(B1,A1) & leftover_mask);



        #ifdef DEBUG
        printf("ACCUM1 TS 1:%d\n", accum1_ts1);

        #endif

        //THRESHOLDING AND BINARIZATION
        *pOut1 |= (accum1_ts1 >= *(thresholds++)) << (32 - output_offset1 -1);

//      asm volatile("":::"memory");

        weights_overstep1  = MODULO_32(weights_overstep1   + dim_ker);

        A1 = *(pA1_next);
        pA1_next =pA1_next + ( ((weights_overstep1 + words_leftover_kernel) >= 32) | dim_ker >= 32);
        weights_mask1 = (1<< (weights_overstep1)) - 1;
        A1 = (A1 << weights_overstep1) |  (*(pA1_next) >> (32-weights_overstep1) & weights_mask1);


        output_offset1+= 1;
        pOut1 += DIVIDE_32(output_offset1);
        output_offset1 =MODULO_32(output_offset1);


    leftover_channels--;
    }


    return pOut1;


}