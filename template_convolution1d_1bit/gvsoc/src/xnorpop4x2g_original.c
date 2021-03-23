#include "rt/rt_api.h"
#include "../include/kernels.h"

//MACROS
#define DIVIDE_32(x) x>>5
#define MODULO_32(x) (x & 0x1F)
#define XNOR(B,A) (~(B ^ A))

#define POPCOUNT(x) __builtin_pulp_cnt(x)

//#define DEBUG


uint32_t *xnorpop4x2(
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

//DECLARE VARIABLES
//    uint32_t *pB1;
    uint32_t *pB1_next, B1;
//    uint32_t *pB2;

    uint32_t input_offset2;

    uint32_t *pA1, *pA1_next;
    uint32_t A1;


    uint32_t *pOut2;

    //Variables
    //NxM dependant
    //N
    uint32_t leftover_channels = ch_out % 4;
    //M
    input_offset2 = MODULO_32(input_offset1 + ch_in);
    //Other variables
    //uint32_t dim_ker>=32  = dim_ker>=32;
    uint32_t words_kernel = DIVIDE_32(dim_ker);
    uint32_t words_leftover_kernel = MODULO_32(dim_ker);
    uint32_t leftover_mask = words_leftover_kernel != 0 ? ~((1 << (32 - words_leftover_kernel)) - 1) : 0x0;

    uint32_t weights_overstep1 = 0;
    uint32_t weights_mask1 = (1 << (weights_overstep1)) - 1;


    uint32_t output_offset2 = MODULO_32(output_offset1 + ch_out);


    pA1 = pWeight;
    pA1_next = pWeight + (dim_ker >= 32);
    pOut2 = pOut1 + ((output_offset1 + ch_out) >> 5);

    //Load
    A1 = *(pA1);

    //Iterazioni sui canali in output
    int out_channel = 0;
    if((ch_out>>2) >0) {
    do {
//    for(int out_channel=0; out_channel< (ch_out)>>2; out_channel++) {
#ifdef DEBUG
        printf("-------------------------\n");
#endif
        uint32_t B2;
        //Update pointers and variables
        uint32_t *pB1 = pInBuffer;
        uint32_t *pB2 = pB1 + ((input_offset1 + ch_in) >> 5);

        uint32_t input_mask1 = (1 << input_offset1) - 1;
        uint32_t input_mask2 = (1 << input_offset2) - 1;

        //Additional weights
        uint32_t weights_overstep2 = MODULO_32(weights_overstep1) + words_leftover_kernel;
        uint32_t weights_overstep3 = MODULO_32(weights_overstep2) + words_leftover_kernel;
        uint32_t weights_overstep4 = MODULO_32(weights_overstep3) + words_leftover_kernel;

        uint32_t *pA2 = pA1 + words_kernel + (weights_overstep2 >= 32); //Point pAI all'inizio dei pesi
        uint32_t *pA3 = pA2 + words_kernel + (weights_overstep3 >= 32); //Point pAI all'inizio dei pesi
        uint32_t *pA4 = pA3 + words_kernel + (weights_overstep4 >= 32); //Point pAI all'inizio dei pesi

        weights_overstep2 = MODULO_32(weights_overstep2);
        weights_overstep3 = MODULO_32(weights_overstep3);
        weights_overstep4 = MODULO_32(weights_overstep4);

//        int next_weights_in_next_word1 = ((weights_overstep1 + words_leftover_kernel) >=32);
//        int next_weights_in_next_word2 = ((weights_overstep2 + words_leftover_kernel) >=32);
//        int next_weights_in_next_word3 = ((weights_overstep3 + words_leftover_kernel) >=32);
//        int next_weights_in_next_word4 = ((weights_overstep4 + words_leftover_kernel) >=32);

        uint32_t *pA2_next = pA2 + (((weights_overstep2 + words_leftover_kernel) >= 32) | dim_ker >= 32);
        uint32_t *pA3_next = pA3 + (((weights_overstep3 + words_leftover_kernel) >= 32) | dim_ker >= 32);
        uint32_t *pA4_next = pA4 + (((weights_overstep4 + words_leftover_kernel) >= 32) | dim_ker >= 32);


        uint32_t weights_mask2 = (1 << (weights_overstep2)) - 1;
        uint32_t weights_mask3 = (1 << (weights_overstep3)) - 1;
        uint32_t weights_mask4 = (1 << (weights_overstep4)) - 1;

        //Store the conv output
        int accum1_ts1 = 0;
        int accum2_ts1 = 0;
        int accum3_ts1 = 0;
        int accum4_ts1 = 0;
        int accum1_ts2 = 0;
        int accum2_ts2 = 0;
        int accum3_ts2 = 0;
        int accum4_ts2 = 0;

        //LOAD

//        B1 = *(pB1++);
//        B2 = *(pB2++);

//        B1_next = *(pB1);
//        B2_next = *(pB2);

        uint32_t A2 = *(pA2);
        uint32_t A3 = *(pA3);
        uint32_t A4 = *(pA4);

        //Operations on loaded values
//        B1 <<= input_offset1;
        B1 = ((*(pB1++)) << input_offset1) | (*(pB1) >> (32 - input_offset1) & input_mask1);
//        B2 <<= input_offset2;
        B2 = ((*(pB2++)) << input_offset2) | (*(pB2) >> (32 - input_offset2) & input_mask2);


        A2 = A2 << weights_overstep2 | (*(pA2_next) >> (32 - weights_overstep2) & weights_mask2);
        A3 = A3 << weights_overstep3 | (*(pA3_next) >> (32 - weights_overstep3) & weights_mask3);
        A4 = A4 << weights_overstep4 | (*(pA4_next) >> (32 - weights_overstep4) & weights_mask4);

//        int words_in =0;
//        if(words_kernel>0){
//            do{
//
//
//                words_in++;
//            } while(words_in<words_kernel);
//
//        }
        if((words_kernel)>0) {
            for (int words_in = 0; words_in < (words_kernel - 1); words_in++) {

                accum1_ts1 += POPCOUNT(XNOR(B1, A1));
                accum2_ts1 += POPCOUNT(XNOR(B1, A2));
                accum3_ts1 += POPCOUNT(XNOR(B1, A3));
                accum4_ts1 += POPCOUNT(XNOR(B1, A4));
                accum1_ts2 += POPCOUNT(XNOR(B2, A1));
                accum2_ts2 += POPCOUNT(XNOR(B2, A2));
                accum3_ts2 += POPCOUNT(XNOR(B2, A3));
                accum4_ts2 += POPCOUNT(XNOR(B2, A4));
//            asm volatile("":::"memory");

#ifdef DEBUG
                printf("A1:%x\n",A1);
                printf("A2:%x\n",A2);
                printf("A3:%x\n",A3);
                printf("A4:%x\n",A4);
                printf("B1:%x\n",B1);
                printf("B2:%x\n",B2);
#endif

                //Swap values
//            B1 = *(pB1++);
//            B2 = *(pB2++);
                A1 = *(pA1_next);
                A2 = *(pA2_next);
                A3 = *(pA3_next);
                A4 = *(pA4_next);


//            B1_next = *(++pB1);
//            B2_next = *(++pB2);


//            B1 <<= input_offset1;
                B1 = ((*(pB1++)) << input_offset1) | ((*(pB1) >> (32 - input_offset1)) & input_mask1);
//            B2 <<= input_offset2;
                B2 = ((*(pB2++)) << input_offset2) | ((*(pB2) >> (32 - input_offset2)) & input_mask2);
                A1 <<= weights_overstep1;
                A2 <<= weights_overstep2;
                A3 <<= weights_overstep3;
                A4 <<= weights_overstep4;

                //int increment_flag = ; //1 if it's not the last iteration
                pA1_next += 1;
                pA2_next += 1;
                pA3_next += 1;
                pA4_next += 1;


                A1 = A1 | ((*pA1_next >> (32 - weights_overstep1)) & weights_mask1);
                A2 = A2 | ((*pA2_next >> (32 - weights_overstep2)) & weights_mask2);
                A3 = A3 | ((*pA3_next >> (32 - weights_overstep3)) & weights_mask3);
                A4 = A4 | ((*pA4_next >> (32 - weights_overstep4)) & weights_mask4);

            }

            //Last iteration here
            accum1_ts1 += POPCOUNT(XNOR(B1, A1));
            accum2_ts1 += POPCOUNT(XNOR(B1, A2));
            accum3_ts1 += POPCOUNT(XNOR(B1, A3));
            accum4_ts1 += POPCOUNT(XNOR(B1, A4));
            accum1_ts2 += POPCOUNT(XNOR(B2, A1));
            accum2_ts2 += POPCOUNT(XNOR(B2, A2));
            accum3_ts2 += POPCOUNT(XNOR(B2, A3));
            accum4_ts2 += POPCOUNT(XNOR(B2, A4));


            A1 = *(pA1_next);
            A2 = *(pA2_next);
            A3 = *(pA3_next);
            A4 = *(pA4_next);


//            B1_next = *(++pB1);
//            B2_next = *(++pB2);


//            B1 <<= input_offset1;
            B1 = ((*(pB1++)) << input_offset1) | ((*(pB1) >> (32 - input_offset1)) & input_mask1);
//            B2 <<= input_offset2;
            B2 = ((*(pB2++)) << input_offset2) | ((*(pB2) >> (32 - input_offset2)) & input_mask2);
            A1 <<= weights_overstep1;
            A2 <<= weights_overstep2;
            A3 <<= weights_overstep3;
            A4 <<= weights_overstep4;

            //int increment_flag = ; //1 if it's not the last iteration
            pA1_next += ((weights_overstep1 + words_leftover_kernel) >= 32);
            pA2_next += ((weights_overstep2 + words_leftover_kernel) >= 32);
            pA3_next += ((weights_overstep3 + words_leftover_kernel) >= 32);
            pA4_next += ((weights_overstep4 + words_leftover_kernel) >= 32);


            A1 = A1 | ((*pA1_next >> (32 - weights_overstep1)) & weights_mask1);
            A2 = A2 | ((*pA2_next >> (32 - weights_overstep2)) & weights_mask2);
            A3 = A3 | ((*pA3_next >> (32 - weights_overstep3)) & weights_mask3);
            A4 = A4 | ((*pA4_next >> (32 - weights_overstep4)) & weights_mask4);
        }

        //ODD KERNELS or DIM_KER < 32
#ifdef DEBUG
        printf("A1:%x\n", A1 & leftover_mask);
        printf("A2:%x\n", A2 & leftover_mask);
        printf("A3:%x\n", A3 & leftover_mask);
        printf("A4:%x\n", A4 & leftover_mask);
        printf("B1:%x\n",B1 & leftover_mask);
        printf("B2:%x\n",B2 & leftover_mask);
#endif
        //if(words_leftover_kernel) {
        accum1_ts1 += POPCOUNT(XNOR(B1, A1) & leftover_mask);
        accum2_ts1 += POPCOUNT(XNOR(B1, A2) & leftover_mask);
        accum3_ts1 += POPCOUNT(XNOR(B1, A3) & leftover_mask);
        accum4_ts1 += POPCOUNT(XNOR(B1, A4) & leftover_mask);
        accum1_ts2 += POPCOUNT(XNOR(B2, A1) & leftover_mask);
        accum2_ts2 += POPCOUNT(XNOR(B2, A2) & leftover_mask);
        accum3_ts2 += POPCOUNT(XNOR(B2, A3) & leftover_mask);
        accum4_ts2 += POPCOUNT(XNOR(B2, A4) & leftover_mask);
        //}


#ifdef DEBUG
        printf("ACCUM1 TS 1:%d\n", accum1_ts1);
        printf("ACCUM2 TS 1:%d\n", accum2_ts1);
        printf("ACCUM3 TS 1:%d\n", accum3_ts1);
        printf("ACCUM4 TS 1:%d\n", accum4_ts1);
        printf("ACCUM1 TS 2:%d\n", accum1_ts2);
        printf("ACCUM2 TS 2:%d\n", accum2_ts2);
        printf("ACCUM3 TS 2:%d\n", accum3_ts2);
        printf("ACCUM4 TS 2:%d\n", accum4_ts2);

#endif

        //THRESHOLDING AND BINARIZATION
        *pOut1 |= (accum1_ts1 >= *(thresholds)) << (32 - output_offset1 - 1);
        *pOut2 |= (accum1_ts2 >= *(thresholds++)) << (32 - output_offset2 - 1);

        *pOut1 |= (accum2_ts1 >= *(thresholds)) << (32 - output_offset1 - 2);
        *pOut2 |= (accum2_ts2 >= *(thresholds++)) << (32 - output_offset2 - 2);
        *pOut1 |= (accum3_ts1 >= *(thresholds)) << (32 - output_offset1 - 3);
        *pOut2 |= (accum3_ts2 >= *(thresholds++)) << (32 - output_offset2 - 3);
        *pOut1 |= (accum4_ts1 >= *(thresholds)) << (32 - output_offset1 - 4);
        *pOut2 |= (accum4_ts2 >= *(thresholds++)) << (32 - output_offset2 - 4);

        weights_overstep1 = MODULO_32(weights_overstep4 + dim_ker);

        A1 = *(pA4_next);
//        A1 <<= weights_overstep1;
        //int next_weights_in_next_word = ;
        pA1_next = pA4_next + ( ((weights_overstep1 + words_leftover_kernel) >= 32) | dim_ker >= 32);
        weights_mask1 = (1 << (weights_overstep1)) - 1;
        A1 = (A1 << weights_overstep1) | (*(pA1_next) >> (32 - weights_overstep1) & weights_mask1);

        pA1 = pA4_next;

        output_offset1 += 4;
        pOut1 += output_offset1 >> 5;
        output_offset1 &= 31;
        output_offset2 += 4;
        pOut2 += output_offset2 >> 5;
        output_offset2 &= 31;

        out_channel++;
    } while (out_channel < (ch_out) >> 2);
}


    return pOut2;


}