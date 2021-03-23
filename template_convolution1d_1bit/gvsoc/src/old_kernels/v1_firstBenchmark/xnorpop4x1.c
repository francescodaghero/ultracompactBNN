#include "rt/rt_api.h"
#include "../include/kernels.h"

//MACROS
#define DIVIDE_32(x) x>>5
#define MODULO_32(x) (x & 0x1F)
#define XNOR(B,A) (~(B ^ A))

#define POPCOUNT(x) __builtin_pulp_cnt(x)

//#define DEBUG


uint32_t *xnorpop4x1(
    const uint32_t * pInBuffer,
    const uint32_t input_offset1,
    const uint16_t  dim_in,
    const uint16_t  ch_in,
    const uint32_t *  pWeight,
    const uint16_t  ch_out,
    const uint16_t  kernel_size,
    const uint16_t  stride,
    uint32_t *       pOut1,
    uint32_t output_offset1,
    const uint16_t  dim_out,
    const int16_t *thresholds
) {

//DECLARE VARIABLES
    uint32_t *pB1;
    uint32_t *pB1_next, B1, B1_next;

    uint32_t *pA1, *pA1_next;
    uint32_t A1, A1_next;
    uint32_t *pA2, *pA2_next;
    uint32_t A2, A2_next;
    uint32_t weights_mask2;
    uint32_t *pA3, *pA3_next;
    uint32_t A3, A3_next;
    uint32_t weights_mask3;
    uint32_t *pA4, *pA4_next;
    uint32_t A4, A4_next;
    uint32_t weights_mask4;


    //Variables
    //NxM dependant
    //N
    uint32_t leftover_channels = ch_out %4;
    //M
    //Other variables
    int dim_ker=ch_in*kernel_size; //Costante, posso calcolarla offline
    uint32_t next_pA = dim_ker>=32;
    uint32_t words_kernel = DIVIDE_32(dim_ker);
    uint32_t words_leftover_kernel = MODULO_32(dim_ker);
    uint32_t leftover_mask = words_leftover_kernel!=0 ? ~((1 << (32 - words_leftover_kernel )) - 1) : 0x0;

    uint32_t weights_overstep1 =0;
    uint32_t weights_mask1 = (1<< (weights_overstep1)) - 1;


    //Variabile per l'output
    uint32_t z1 = 0;
    uint32_t savecounter1 = output_offset1;


    //Update Pointers
    pB1 = pInBuffer;

    pA1 = pWeight;
    pA1_next = pWeight + (next_pA);

    //Load
    A1 = *(pA1);
    A1_next = *(pA1_next);

    //Iterazioni sui canali in output
    for(int out_channel=0; out_channel< (ch_out)>>2; out_channel++) {
        #ifdef DEBUG
        printf("-------------------------\n");
        #endif

        //Update pointers and variables
        pB1 = pInBuffer;

        uint32_t input_mask1 = (1<< input_offset1) - 1;

        //Additional weights
        uint32_t weights_overstep2 = MODULO_32(weights_overstep1)+words_leftover_kernel;
        uint32_t weights_overstep3 = MODULO_32(weights_overstep2)+words_leftover_kernel;
        uint32_t weights_overstep4 = MODULO_32(weights_overstep3)+words_leftover_kernel;

        pA2 = pA1 + words_kernel + (weights_overstep2 >=32); //Point pAI all'inizio dei pesi
        pA3 = pA2 + words_kernel + (weights_overstep3 >=32); //Point pAI all'inizio dei pesi
        pA4 = pA3 + words_kernel + (weights_overstep4 >=32); //Point pAI all'inizio dei pesi

        weights_overstep2 = MODULO_32(weights_overstep2);
        weights_overstep3 = MODULO_32(weights_overstep3);
        weights_overstep4 = MODULO_32(weights_overstep4);

        int next_weights_in_next_word1 = ((weights_overstep1 + words_leftover_kernel) >=32);
        int next_weights_in_next_word2 = ((weights_overstep2 + words_leftover_kernel) >=32);
        int next_weights_in_next_word3 = ((weights_overstep3 + words_leftover_kernel) >=32);
        int next_weights_in_next_word4 = ((weights_overstep4 + words_leftover_kernel) >=32);

        pA2_next = pA2 + (next_weights_in_next_word2 | next_pA);
        pA3_next = pA3 + (next_weights_in_next_word3 | next_pA);
        pA4_next = pA4 + (next_weights_in_next_word4 | next_pA);

        weights_mask2 = (1<< (weights_overstep2)) - 1;
        weights_mask3 = (1<< (weights_overstep3)) - 1;
        weights_mask4 = (1<< (weights_overstep4)) - 1;

        //Store the conv output
        int accum1_ts1 =0;
        int accum2_ts1 =0;
        int accum3_ts1 =0;
        int accum4_ts1 =0;

        //LOAD

        B1 = (*pB1);

        B1_next = *(++pB1);

        A2 = *(pA2);
        A3 = *(pA3);
        A4 = *(pA4);

        A2_next = *(pA2_next);
        A3_next = *(pA3_next);
        A4_next = *(pA4_next);

        //Operations on loaded values
        B1 <<= input_offset1;
        B1= B1| (B1_next >> (32  - input_offset1) & input_mask1);


        A2 = A2 << weights_overstep2 |  (A2_next >> (32-weights_overstep2) & weights_mask2);
        A3 = A3 << weights_overstep3 |  (A3_next >> (32-weights_overstep3) & weights_mask3);
        A4 = A4 << weights_overstep4 |  (A4_next >> (32-weights_overstep4) & weights_mask4);


        for (int words_in = 0; words_in < words_kernel; words_in++) {

            accum1_ts1 += POPCOUNT(XNOR(B1,A1));
            accum2_ts1 += POPCOUNT(XNOR(B1,A2));
            accum3_ts1 += POPCOUNT(XNOR(B1,A3));
            accum4_ts1 += POPCOUNT(XNOR(B1,A4));

            asm volatile("":::"memory");

#ifdef DEBUG
            printf("A1:%x\n",A1);
            printf("A2:%x\n",A2);
            printf("A3:%x\n",A3);
            printf("A4:%x\n",A4);
            printf("B1:%x\n",B1);
            #endif

            //Swap values
            B1 = B1_next;
            A1 = A1_next;
            A2 = A2_next;
            A3 = A3_next;
            A4 = A4_next;

            B1_next = *(++pB1);

            B1 <<= input_offset1;
            B1= B1 | ( (B1_next >> (32  - input_offset1)) & input_mask1);
            A1 <<= weights_overstep1;
            A2 <<= weights_overstep2;
            A3 <<= weights_overstep3;
            A4 <<= weights_overstep4;

            int increment_flag = ( words_in != (words_kernel - 1)); //1 if it's not the last iteration
            pA1_next += next_weights_in_next_word1| increment_flag;
            pA2_next += next_weights_in_next_word2| increment_flag;
            pA3_next += next_weights_in_next_word3| increment_flag;
            pA4_next += next_weights_in_next_word4| increment_flag;

            A1_next = *(pA1_next);
            A2_next = *(pA2_next);
            A3_next = *(pA3_next);
            A4_next = *(pA4_next);

            A1 = A1 | ( (A1_next >> (32-weights_overstep1 )) & weights_mask1);
            A2 = A2 | ( (A2_next >> (32-weights_overstep2 )) & weights_mask2);
            A3 = A3 | ( (A3_next >> (32-weights_overstep3 )) & weights_mask3);
            A4 = A4 | ( (A4_next >> (32-weights_overstep4 )) & weights_mask4);

        }


        //ODD KERNELS or DIM_KER < 32
        #ifdef DEBUG
        printf("A1:%x\n", A1 & leftover_mask);
        printf("A2:%x\n", A2 & leftover_mask);
        printf("A3:%x\n", A3 & leftover_mask);
        printf("A4:%x\n", A4 & leftover_mask);
        printf("B1:%x\n",B1 & leftover_mask);
        #endif
        accum1_ts1 += POPCOUNT(XNOR(B1,A1) & leftover_mask);
        accum2_ts1 += POPCOUNT(XNOR(B1,A2) & leftover_mask);
        accum3_ts1 += POPCOUNT(XNOR(B1,A3) & leftover_mask);
        accum4_ts1 += POPCOUNT(XNOR(B1,A4) & leftover_mask);



        #ifdef DEBUG
        printf("ACCUM1 TS 1:%d\n", accum1_ts1);
        printf("ACCUM2 TS 1:%d\n", accum2_ts1);
        printf("ACCUM3 TS 1:%d\n", accum3_ts1);
        printf("ACCUM4 TS 1:%d\n", accum4_ts1);

        #endif

        //THRESHOLDING AND BINARIZATION
        uint32_t tmp1;

        tmp1 = accum1_ts1 >= *(thresholds++);

        tmp1 = (tmp1 << 1) + (accum2_ts1 >= *(thresholds++));
        tmp1 = (tmp1 << 1) + (accum3_ts1 >= *(thresholds++));
        tmp1 = (tmp1 << 1) + (accum4_ts1 >= *(thresholds++));

        z1 |= tmp1 << (32 - savecounter1 - 4);

        weights_overstep1  = MODULO_32(weights_overstep4   + dim_ker);

        A1 = A4_next;
        A1 <<= weights_overstep1;
        int next_weights_in_next_word = ((weights_overstep1 + words_leftover_kernel) >=32) | next_pA;
        pA1_next =pA4_next + next_weights_in_next_word;
        A1_next = *(pA1_next);
        weights_mask1 = (1<< (weights_overstep1)) - 1;
        A1 = A1 |  (A1_next >> (32-weights_overstep1) & weights_mask1);

        pA1 = pA4_next;

        savecounter1 +=4;
        if((savecounter1)%32==0) {
            *(pOut1++)|=(z1);
            savecounter1=0;
            z1 = 0;
        }

    }

    while(leftover_channels) {
    #ifdef DEBUG
        printf("-------------------------\n");
        #endif

        //Update pointers and variables
        pB1 = pInBuffer;

        uint32_t input_mask1 = (1<< input_offset1) - 1;

        //Additional weights



        int next_weights_in_next_word1 = ((weights_overstep1 + words_leftover_kernel) >=32);



        //Store the conv output
        int accum1_ts1 =0;

        //LOAD

        B1 = (*pB1);

        B1_next = *(++pB1);



        //Operations on loaded values
        B1 <<= input_offset1;
        B1= B1| (B1_next >> (32  - input_offset1) & input_mask1);




        for (int words_in = 0; words_in < words_kernel; words_in++) {

            accum1_ts1 += POPCOUNT(XNOR(B1,A1));


            #ifdef DEBUG
            printf("A1:%x\n",A1);
            printf("B1:%x\n",B1);
            #endif

            //Swap values
            B1 = B1_next;
            A1 = A1_next;

            B1_next = *(++pB1);

            B1 <<= input_offset1;
            B1= B1 | ( (B1_next >> (32  - input_offset1)) & input_mask1);
            A1 <<= weights_overstep1;

            int increment_flag = ( words_in != (words_kernel - 1)); //1 if it's not the last iteration
            pA1_next += next_weights_in_next_word1| increment_flag;

            A1_next = *(pA1_next);

            A1 = A1 | ( (A1_next >> (32-weights_overstep1 )) & weights_mask1);

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
        uint32_t tmp1;

        tmp1 = accum1_ts1 >= *(thresholds++);


        z1 |= tmp1 << (32 - savecounter1 - 1);

        weights_overstep1  = MODULO_32(weights_overstep1   + dim_ker);

        A1 = A1_next;
        A1 <<= weights_overstep1;
        int next_weights_in_next_word = ((weights_overstep1 + words_leftover_kernel) >=32) | next_pA;
        pA1_next =pA1_next + next_weights_in_next_word;
        A1_next = *(pA1_next);
        weights_mask1 = (1<< (weights_overstep1)) - 1;
        A1 = A1 |  (A1_next >> (32-weights_overstep1) & weights_mask1);


        savecounter1 +=1;
        if((savecounter1)%32==0) {
            *(pOut1++)|=(z1);
            savecounter1=0;
            z1 = 0;
        }


    leftover_channels--;
    }

    if(z1!=0)
        {
            *(pOut1) |= (z1);
    pOut1 += ((output_offset1+ch_out)==32);
    }


    return pOut1;


}