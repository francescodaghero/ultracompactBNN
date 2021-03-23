#include "rt/rt_api.h"
#include "../include/kernels.h"

//MACROS
#define DIVIDE_32(x) x>>5
#define MODULO_32(x) (x & 0x1F)
#define XNOR(B,A) (~(B ^ A))

#define POPCOUNT(x) __builtin_pulp_cnt(x)

//#define DEBUG


uint32_t *xnorpop2x4(
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
    uint32_t *pB2;
    uint32_t *pB2_next, B2, B2_next;
    uint32_t input_offset2;
    uint32_t *pB3;
    uint32_t *pB3_next, B3, B3_next;
    uint32_t input_offset3;
    uint32_t *pB4;
    uint32_t *pB4_next, B4, B4_next;
    uint32_t input_offset4;

    uint32_t *pA1, *pA1_next;
    uint32_t A1, A1_next;
    uint32_t *pA2, *pA2_next;
    uint32_t A2, A2_next;
    uint32_t weights_mask2;

    uint32_t *pOut2;
    uint32_t *pOut3;
    uint32_t *pOut4;

    //Variables
    //NxM dependant
    //N
    uint32_t leftover_channels = ch_out %2;
    //M
    input_offset2 = MODULO_32(input_offset1 + ch_in);
    input_offset3 = MODULO_32(input_offset2 + ch_in);
    input_offset4 = MODULO_32(input_offset3 + ch_in);
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

    uint32_t output_offset2 = MODULO_32(output_offset1 + ch_out);
    uint32_t z2 = 0;
    uint32_t savecounter2 = output_offset2;
    uint32_t output_offset3 = MODULO_32(output_offset2 + ch_out);
    uint32_t z3 = 0;
    uint32_t savecounter3 = output_offset3;
    uint32_t output_offset4 = MODULO_32(output_offset3 + ch_out);
    uint32_t z4 = 0;
    uint32_t savecounter4 = output_offset4;

    //Update Pointers
    pB1 = pInBuffer;
    pB2 = pInBuffer + ((input_offset1 + ch_in)>>5);
    pB3 = pInBuffer + ((input_offset2 + ch_in)>>5);
    pB4 = pInBuffer + ((input_offset3 + ch_in)>>5);

    pA1 = pWeight;
    pA1_next = pWeight + (next_pA);
    pOut2 = pOut1 + ((output_offset1 + ch_out)>>5);
    pOut3 = pOut2 + ((output_offset2 + ch_out)>>5);
    pOut4 = pOut3 + ((output_offset3 + ch_out)>>5);

    //Load
    A1 = *(pA1);
    A1_next = *(pA1_next);

    //Iterazioni sui canali in output
    for(int out_channel=0; out_channel< (ch_out)>>1; out_channel++) {
        #ifdef DEBUG
        printf("-------------------------\n");
        #endif

        //Update pointers and variables
        pB1 = pInBuffer;
        pB2 = pB1  + ((input_offset1 + ch_in)>>5);
        pB3 = pB2  + ((input_offset2 + ch_in)>>5);
        pB4 = pB3  + ((input_offset3 + ch_in)>>5);

        uint32_t input_mask1 = (1<< input_offset1) - 1;
        uint32_t input_mask2 = (1<< input_offset2) - 1;
        uint32_t input_mask3 = (1<< input_offset3) - 1;
        uint32_t input_mask4 = (1<< input_offset4) - 1;

        //Additional weights
        uint32_t weights_overstep2 = MODULO_32(weights_overstep1)+words_leftover_kernel;

        pA2 = pA1 + words_kernel + (weights_overstep2 >=32); //Point pAI all'inizio dei pesi

        weights_overstep2 = MODULO_32(weights_overstep2);

        int next_weights_in_next_word1 = ((weights_overstep1 + words_leftover_kernel) >=32);
        int next_weights_in_next_word2 = ((weights_overstep2 + words_leftover_kernel) >=32);

        pA2_next = pA2 + (next_weights_in_next_word2 | next_pA);

        weights_mask2 = (1<< (weights_overstep2)) - 1;

        //Store the conv output
        int accum1_ts1 =0;
        int accum1_ts2 =0;
        int accum1_ts3 =0;
        int accum1_ts4 =0;
        int accum2_ts1 =0;
        int accum2_ts2 =0;
        int accum2_ts3 =0;
        int accum2_ts4 =0;

        //LOAD

        B1 = (*pB1);
        B2 = (*pB2);
        B3 = (*pB3);
        B4 = (*pB4);

        B1_next = *(++pB1);
        B2_next = *(++pB2);
        B3_next = *(++pB3);
        B4_next = *(++pB4);

        A2 = *(pA2);

        A2_next = *(pA2_next);

        //Operations on loaded values
        B1 <<= input_offset1;
        B1= B1| (B1_next >> (32  - input_offset1) & input_mask1);
        B2 <<= input_offset2;
        B2= B2| (B2_next >> (32  - input_offset2) & input_mask2);
        B3 <<= input_offset3;
        B3= B3| (B3_next >> (32  - input_offset3) & input_mask3);
        B4 <<= input_offset4;
        B4= B4| (B4_next >> (32  - input_offset4) & input_mask4);


        A2 = A2 << weights_overstep2 |  (A2_next >> (32-weights_overstep2) & weights_mask2);


        for (int words_in = 0; words_in < words_kernel; words_in++) {

            accum1_ts1 += POPCOUNT(XNOR(B1,A1));
            accum1_ts2 += POPCOUNT(XNOR(B2,A1));
            accum1_ts3 += POPCOUNT(XNOR(B3,A1));
            accum1_ts4 += POPCOUNT(XNOR(B4,A1));
            accum2_ts1 += POPCOUNT(XNOR(B1,A2));
            accum2_ts2 += POPCOUNT(XNOR(B2,A2));
            accum2_ts3 += POPCOUNT(XNOR(B3,A2));
            accum2_ts4 += POPCOUNT(XNOR(B4,A2));
            asm volatile("":::"memory");


#ifdef DEBUG
            printf("A1:%x\n",A1);
            printf("A2:%x\n",A2);
            printf("B1:%x\n",B1);
            printf("B2:%x\n",B2);
            printf("B3:%x\n",B3);
            printf("B4:%x\n",B4);
            #endif

            //Swap values
            B1 = B1_next;
            B2 = B2_next;
            B3 = B3_next;
            B4 = B4_next;
            A1 = A1_next;
            A2 = A2_next;

            B1_next = *(++pB1);
            B2_next = *(++pB2);
            B3_next = *(++pB3);
            B4_next = *(++pB4);

            B1 <<= input_offset1;
            B1= B1 | ( (B1_next >> (32  - input_offset1)) & input_mask1);
            B2 <<= input_offset2;
            B2= B2 | ( (B2_next >> (32  - input_offset2)) & input_mask2);
            B3 <<= input_offset3;
            B3= B3 | ( (B3_next >> (32  - input_offset3)) & input_mask3);
            B4 <<= input_offset4;
            B4= B4 | ( (B4_next >> (32  - input_offset4)) & input_mask4);
            A1 <<= weights_overstep1;
            A2 <<= weights_overstep2;

            int increment_flag = ( words_in != (words_kernel - 1)); //1 if it's not the last iteration
            pA1_next += next_weights_in_next_word1| increment_flag;
            pA2_next += next_weights_in_next_word2| increment_flag;

            A1_next = *(pA1_next);
            A2_next = *(pA2_next);

            A1 = A1 | ( (A1_next >> (32-weights_overstep1 )) & weights_mask1);
            A2 = A2 | ( (A2_next >> (32-weights_overstep2 )) & weights_mask2);

        }


        //ODD KERNELS or DIM_KER < 32
        #ifdef DEBUG
        printf("A1:%x\n", A1 & leftover_mask);
        printf("A2:%x\n", A2 & leftover_mask);
        printf("B1:%x\n",B1 & leftover_mask);
        printf("B2:%x\n",B2 & leftover_mask);
        printf("B3:%x\n",B3 & leftover_mask);
        printf("B4:%x\n",B4 & leftover_mask);
        #endif
        accum1_ts1 += POPCOUNT(XNOR(B1,A1) & leftover_mask);
        accum1_ts2 += POPCOUNT(XNOR(B2,A1) & leftover_mask);
        accum1_ts3 += POPCOUNT(XNOR(B3,A1) & leftover_mask);
        accum1_ts4 += POPCOUNT(XNOR(B4,A1) & leftover_mask);
        accum2_ts1 += POPCOUNT(XNOR(B1,A2) & leftover_mask);
        accum2_ts2 += POPCOUNT(XNOR(B2,A2) & leftover_mask);
        accum2_ts3 += POPCOUNT(XNOR(B3,A2) & leftover_mask);
        accum2_ts4 += POPCOUNT(XNOR(B4,A2) & leftover_mask);



        #ifdef DEBUG
        printf("ACCUM1 TS 1:%d\n", accum1_ts1);
        printf("ACCUM1 TS 2:%d\n", accum1_ts2);
        printf("ACCUM1 TS 3:%d\n", accum1_ts3);
        printf("ACCUM1 TS 4:%d\n", accum1_ts4);
        printf("ACCUM2 TS 1:%d\n", accum2_ts1);
        printf("ACCUM2 TS 2:%d\n", accum2_ts2);
        printf("ACCUM2 TS 3:%d\n", accum2_ts3);
        printf("ACCUM2 TS 4:%d\n", accum2_ts4);

        #endif

        //THRESHOLDING AND BINARIZATION
        uint32_t tmp1;
        uint32_t tmp2;
        uint32_t tmp3;
        uint32_t tmp4;

        tmp1 = accum1_ts1 >= *(thresholds);
        tmp2 = accum1_ts2 >= *(thresholds);
        tmp3 = accum1_ts3 >= *(thresholds);
        tmp4 = accum1_ts4 >= *(thresholds++);

        tmp1 = (tmp1 << 1) + (accum2_ts1 >= *(thresholds));
        tmp2 = (tmp2 << 1) + (accum2_ts2 >= *(thresholds));
        tmp3 = (tmp3 << 1) + (accum2_ts3 >= *(thresholds));
        tmp4 = (tmp4 << 1) + (accum2_ts4 >= *(thresholds++));

        z1 |= tmp1 << (32 - savecounter1 - 2);
        z2 |= tmp2 << (32 - savecounter2 - 2);
        z3 |= tmp3 << (32 - savecounter3 - 2);
        z4 |= tmp4 << (32 - savecounter4 - 2);

        weights_overstep1  = MODULO_32(weights_overstep2   + dim_ker);

        A1 = A2_next;
        A1 <<= weights_overstep1;
        int next_weights_in_next_word = ((weights_overstep1 + words_leftover_kernel) >=32) | next_pA;
        pA1_next =pA2_next + next_weights_in_next_word;
        A1_next = *(pA1_next);
        weights_mask1 = (1<< (weights_overstep1)) - 1;
        A1 = A1 |  (A1_next >> (32-weights_overstep1) & weights_mask1);

        pA1 = pA2_next;

        savecounter1 +=2;
        if((savecounter1)%32==0) {
            *(pOut1++)|=(z1);
            savecounter1=0;
            z1 = 0;
        }
        savecounter2 +=2;
        if((savecounter2)%32==0) {
            *(pOut2++)|=(z2);
            savecounter2=0;
            z2 = 0;
        }
        savecounter3 +=2;
        if((savecounter3)%32==0) {
            *(pOut3++)|=(z3);
            savecounter3=0;
            z3 = 0;
        }
        savecounter4 +=2;
        if((savecounter4)%32==0) {
            *(pOut4++)|=(z4);
            savecounter4=0;
            z4 = 0;
        }

    }

    while(leftover_channels) {
    #ifdef DEBUG
        printf("-------------------------\n");
        #endif

        //Update pointers and variables
        pB1 = pInBuffer;
        pB2 = pB1  + ((input_offset1 + ch_in)>>5);
        pB3 = pB2  + ((input_offset2 + ch_in)>>5);
        pB4 = pB3  + ((input_offset3 + ch_in)>>5);

        uint32_t input_mask1 = (1<< input_offset1) - 1;
        uint32_t input_mask2 = (1<< input_offset2) - 1;
        uint32_t input_mask3 = (1<< input_offset3) - 1;
        uint32_t input_mask4 = (1<< input_offset4) - 1;

        //Additional weights



        int next_weights_in_next_word1 = ((weights_overstep1 + words_leftover_kernel) >=32);



        //Store the conv output
        int accum1_ts1 =0;
        int accum1_ts2 =0;
        int accum1_ts3 =0;
        int accum1_ts4 =0;

        //LOAD

        B1 = (*pB1);
        B2 = (*pB2);
        B3 = (*pB3);
        B4 = (*pB4);

        B1_next = *(++pB1);
        B2_next = *(++pB2);
        B3_next = *(++pB3);
        B4_next = *(++pB4);



        //Operations on loaded values
        B1 <<= input_offset1;
        B1= B1| (B1_next >> (32  - input_offset1) & input_mask1);
        B2 <<= input_offset2;
        B2= B2| (B2_next >> (32  - input_offset2) & input_mask2);
        B3 <<= input_offset3;
        B3= B3| (B3_next >> (32  - input_offset3) & input_mask3);
        B4 <<= input_offset4;
        B4= B4| (B4_next >> (32  - input_offset4) & input_mask4);




        for (int words_in = 0; words_in < words_kernel; words_in++) {

            accum1_ts1 += POPCOUNT(XNOR(B1,A1));
            accum1_ts2 += POPCOUNT(XNOR(B2,A1));
            accum1_ts3 += POPCOUNT(XNOR(B3,A1));
            accum1_ts4 += POPCOUNT(XNOR(B4,A1));


            #ifdef DEBUG
            printf("A1:%x\n",A1);
            printf("B1:%x\n",B1);
            printf("B2:%x\n",B2);
            printf("B3:%x\n",B3);
            printf("B4:%x\n",B4);
            #endif

            //Swap values
            B1 = B1_next;
            B2 = B2_next;
            B3 = B3_next;
            B4 = B4_next;
            A1 = A1_next;

            B1_next = *(++pB1);
            B2_next = *(++pB2);
            B3_next = *(++pB3);
            B4_next = *(++pB4);

            B1 <<= input_offset1;
            B1= B1 | ( (B1_next >> (32  - input_offset1)) & input_mask1);
            B2 <<= input_offset2;
            B2= B2 | ( (B2_next >> (32  - input_offset2)) & input_mask2);
            B3 <<= input_offset3;
            B3= B3 | ( (B3_next >> (32  - input_offset3)) & input_mask3);
            B4 <<= input_offset4;
            B4= B4 | ( (B4_next >> (32  - input_offset4)) & input_mask4);
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
        printf("B2:%x\n",B2 & leftover_mask);
        printf("B3:%x\n",B3 & leftover_mask);
        printf("B4:%x\n",B4 & leftover_mask);
        #endif
        accum1_ts1 += POPCOUNT(XNOR(B1,A1) & leftover_mask);
        accum1_ts2 += POPCOUNT(XNOR(B2,A1) & leftover_mask);
        accum1_ts3 += POPCOUNT(XNOR(B3,A1) & leftover_mask);
        accum1_ts4 += POPCOUNT(XNOR(B4,A1) & leftover_mask);



        #ifdef DEBUG
        printf("ACCUM1 TS 1:%d\n", accum1_ts1);
        printf("ACCUM1 TS 2:%d\n", accum1_ts2);
        printf("ACCUM1 TS 3:%d\n", accum1_ts3);
        printf("ACCUM1 TS 4:%d\n", accum1_ts4);

        #endif

        //THRESHOLDING AND BINARIZATION
        uint32_t tmp1;
        uint32_t tmp2;
        uint32_t tmp3;
        uint32_t tmp4;

        tmp1 = accum1_ts1 >= *(thresholds);
        tmp2 = accum1_ts2 >= *(thresholds);
        tmp3 = accum1_ts3 >= *(thresholds);
        tmp4 = accum1_ts4 >= *(thresholds++);


        z1 |= tmp1 << (32 - savecounter1 - 1);
        z2 |= tmp2 << (32 - savecounter2 - 1);
        z3 |= tmp3 << (32 - savecounter3 - 1);
        z4 |= tmp4 << (32 - savecounter4 - 1);

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
        savecounter2 +=1;
        if((savecounter2)%32==0) {
            *(pOut2++)|=(z2);
            savecounter2=0;
            z2 = 0;
        }
        savecounter3 +=1;
        if((savecounter3)%32==0) {
            *(pOut3++)|=(z3);
            savecounter3=0;
            z3 = 0;
        }
        savecounter4 +=1;
        if((savecounter4)%32==0) {
            *(pOut4++)|=(z4);
            savecounter4=0;
            z4 = 0;
        }


    leftover_channels--;
    }

    if(z1!=0)
        {
            *(pOut1) |= (z1);
    }
    if(z2!=0)
        {
            *(pOut2) |= (z2);
    }
    if(z3!=0)
        {
            *(pOut3) |= (z3);
    }
    if(z4!=0)
        {
            *(pOut4) |= (z4);
    pOut4 += ((output_offset4+ch_out)==32);
    }


    return pOut4;


}