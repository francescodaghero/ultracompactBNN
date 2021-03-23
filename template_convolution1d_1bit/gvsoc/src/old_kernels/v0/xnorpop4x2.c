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

//#define DEBUG


static inline uint32_t rotate_right (uint32_t u, size_t r)
{
    __asm__ ("rorl %%cl, %0" : "+r" (u) : "c" (r));
    return u;
}

uint32_t *xnorpop4x2(
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

    //DECLARE VARIABLES
    uint32_t *pB1;
    uint32_t *pB1_next, B1, B1_next;
    uint32_t *pB2;
    uint32_t *pB2_next, B2, B2_next;
    uint32_t input_offset2;

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

    uint32_t *pOut2;

    //LOAD DATA
    pB1 = pInBuffer;
    pB2 = pInBuffer + ((input_offset + 1*ch_in)>>5);

    pA1 = pWeight;
    A1 = (*pA1);
    pA1_next = pWeight + ((ch_in*kernel_size)>=32);
    A1_next = *(pA1_next);
    pOut2 = pOut + ((output_offset + ch_out)>>5);

    //VARIABLES

    //NxM dependant
    //N
    uint32_t leftover_channels = ch_out %4;
    //M
    input_offset2 = MODULO_32(input_offset + ch_in);
    //Other variables
    int dim_ker=ch_in*kernel_size; //Costante, posso calcolarla offline
    uint32_t next_pA = dim_ker>=32;
    uint32_t words_kernel = DIVIDE_32(dim_ker);
    uint32_t words_leftover_kernel = MODULO_32(dim_ker);
    uint32_t leftover_mask = words_leftover_kernel!=0 ? ~((1 << (32 - words_leftover_kernel )) - 1) : 0x0;

    uint32_t weights_overstep1 =0;
    uint32_t weights_mask1 = (1<< (weights_overstep1)) - 1;


    //Variabile per l'output
    uint32_t z = 0;
    uint32_t savecounter = output_offset;

    //Variabile per l'output
    uint32_t output_offset2 = MODULO_32(output_offset + ch_out);
    uint32_t z2 = 0;
    uint32_t savecounter2 = output_offset2;


    //Iterazioni sui canali in output
    for(int out_channel=0; out_channel< (ch_out>>2); out_channel++) {

        #ifdef DEBUG
            printf("-------------------------\n");
        #endif


        pB1 = pInBuffer;
        pB2 = pInBuffer  + ((input_offset + ch_in)>>5);
        B1 = (*pB1);
        B2 = (*pB2);
        B1_next = *(++pB1);
        B2_next = *(++pB2);

        uint32_t input_mask = (1<< input_offset) - 1;
        uint32_t input_mask2 = (1<< input_offset2) - 1;

        B1 <<= input_offset;
        B1=  B1| (B1_next >> (32  - input_offset) & input_mask);
        B2 <<= input_offset2;
        B2= B2| (B2_next >> (32  - input_offset2) & input_mask2);


        //Additional weights
        uint32_t weights_overstep2 = MODULO_32(weights_overstep1)+words_leftover_kernel;
        uint32_t weights_overstep3 = MODULO_32(weights_overstep2)+words_leftover_kernel;
        uint32_t weights_overstep4 = MODULO_32(weights_overstep3)+words_leftover_kernel;


        pA2 = pA1 + words_kernel + (weights_overstep2 >=32); //Point pA2 all'inizio dei pesi
        pA3 = pA2 + words_kernel + (weights_overstep3 >=32); //Point pA3 all'inizio dei pesi
        pA4 = pA3 + words_kernel + (weights_overstep4 >=32); //Point pA4 all'inizio dei pesi

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

        A2 = *(pA2);
        A3 = *(pA3);
        A4 = *(pA4);

        A2_next = *(pA2_next);
        A3_next = *(pA3_next);
        A4_next = *(pA4_next);

        weights_mask2 = (1<< (weights_overstep2)) - 1;
        weights_mask3 = (1<< (weights_overstep3)) - 1;
        weights_mask4 = (1<< (weights_overstep4)) - 1;

        A2 = A2 << weights_overstep2 |  (A2_next >> (32-weights_overstep2) & weights_mask2);
        A3 = A3 << weights_overstep3 |  (A3_next >> (32-weights_overstep3) & weights_mask3);
        A4 = A4 << weights_overstep4 |  (A4_next >> (32-weights_overstep4) & weights_mask4);


        //Store the conv output
        int accum1_ts0 =0;
        int accum1_ts1 =0;
        int accum2_ts0=0;
        int accum2_ts1=0;
        int accum3_ts0=0;
        int accum3_ts1=0;
        int accum4_ts0=0;
        int accum4_ts1=0;

        for (int words_in = 0; words_in < words_kernel; words_in++) {

            accum1_ts0 += POPCOUNT(XNOR(B1,A1));
            accum2_ts0 += POPCOUNT(XNOR(B1,A2));
            accum3_ts0 += POPCOUNT(XNOR(B1,A3));
            accum4_ts0 += POPCOUNT(XNOR(B1,A4));

            accum1_ts1 += POPCOUNT(XNOR(B2,A1));
            accum2_ts1 += POPCOUNT(XNOR(B2,A2));
            accum3_ts1 += POPCOUNT(XNOR(B2,A3));
            accum4_ts1 += POPCOUNT(XNOR(B2,A4));

            #ifdef DEBUG
                printf("A:%x\n", A1);
                printf("A2:%x\n", A2);
                printf("A3:%x\n", A3);
                printf("A4:%x\n", A4);
                printf("B1:%x\n",B1);
                printf("B2:%x\n",B2);

                #endif

            //Swap values
            B1 = B1_next;
            B2 = B2_next;

            A1 = A1_next;
            A2 = A2_next;
            A3 = A3_next;
            A4 = A4_next;


            B1_next = *(++pB1);
            B2_next = *(++pB2);

            B1 <<= input_offset;
            B1= B1 | ( (B1_next >> (32  - input_offset)) & input_mask);

            B2 <<= input_offset2;
            B2= (B2) | ( (B2_next >> (32  - input_offset2)) & input_mask2);

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

            //asm volatile("":::"memory");


            A1 = A1 | ( (A1_next >> (32-weights_overstep1 )) & weights_mask1);
            A2 = A2 | ( (A2_next >> (32-weights_overstep2)) & weights_mask2);
            A3 = A3 | ( (A3_next >> (32-weights_overstep3)) & weights_mask3);
            A4 = A4 | ( (A4_next >> (32-weights_overstep4)) & weights_mask4);

        }


        //ODD KERNELS or DIM_KER < 32
#ifdef DEBUG
            printf("A:%x\n", A1 & leftover_mask);
            printf("A2:%x\n", A2 & leftover_mask);
            printf("A3:%x\n", A3 & leftover_mask);
            printf("A4:%x\n", A4 & leftover_mask);
            printf("B:%x\n",B1 & leftover_mask);
            printf("B2:%x\n",B2 & leftover_mask);
#endif
        accum1_ts0 += POPCOUNT(XNOR(B1,A1) & leftover_mask);
        accum2_ts0 += POPCOUNT(XNOR(B1,A2) & leftover_mask);
        accum3_ts0 += POPCOUNT(XNOR(B1,A3) & leftover_mask);
        accum4_ts0 += POPCOUNT(XNOR(B1,A4) & leftover_mask);

        accum1_ts1 += POPCOUNT(XNOR(B2,A1) & leftover_mask);
        accum2_ts1 += POPCOUNT(XNOR(B2,A2) & leftover_mask);
        accum3_ts1 += POPCOUNT(XNOR(B2,A3) & leftover_mask);
        accum4_ts1 += POPCOUNT(XNOR(B2,A4) & leftover_mask);


#ifdef DEBUG
        printf("ACCUM:%d\n", accum1_ts0);
            printf("ACCUM2:%d\n", accum2_ts0);
            printf("ACCUM3:%d\n", accum3_ts0);
            printf("ACCUM4:%d\n", accum4_ts0);
            printf("ACCUM:%d\n", accum1_ts1);
            printf("ACCUM2:%d\n", accum2_ts1);
            printf("ACCUM3:%d\n", accum3_ts1);
            printf("ACCUM4:%d\n", accum4_ts1);
#endif

        //THRESHOLDING AND BINARIZATION
        /*int16_t thresholds1 = *(thresholds++);
        int16_t thresholds2 = *(thresholds++);
        int16_t thresholds3 = *(thresholds++);
        int16_t thresholds4 = *(thresholds++);
        asm volatile("":::"memory");

        uint32_t tmp, tmp2;

        tmp = accum1_ts0 >= thresholds1;
        tmp2 = accum1_ts1 >= thresholds1;
        tmp = (tmp << 1) + (accum2_ts0 >= thresholds2);
        tmp2 = (tmp2 << 1) + (accum2_ts1 >= thresholds2);
        tmp = (tmp << 1) + (accum3_ts0 >= thresholds3);
        tmp2 = (tmp2 << 1) + (accum3_ts1 >= thresholds3);
        tmp = (tmp << 1) + (accum4_ts0 >= thresholds4);
        tmp2 = (tmp2 << 1) + (accum4_ts1 >= thresholds4);*/

        //THRESHOLDING AND BINARIZATION
        uint32_t tmp, tmp2;
        tmp = accum1_ts0 >= *(thresholds);
        tmp2 = accum1_ts1 >= *(thresholds++);
        tmp = (tmp << 1) + (accum2_ts0 >= *(thresholds));
        tmp2 = (tmp2 << 1) + (accum2_ts1 >= *(thresholds++));
        tmp = (tmp << 1) + (accum3_ts0 >= *(thresholds));
        tmp2 = (tmp2 << 1) + (accum3_ts1 >= *(thresholds++));
        tmp = (tmp << 1) + (accum4_ts0 >= *(thresholds));
        tmp2 = (tmp2 << 1) + (accum4_ts1 >= *(thresholds++));

        /*uint32_t tmp, tmp2;
        int16_t *start_th = thresholds;
        tmp =  ( (accum1_ts0 >= *(thresholds++)) <<3) +
               ((accum2_ts0 >= *(thresholds++))<<2) +
               ((accum3_ts0 >= *(thresholds++))<<1) +
               ((accum4_ts0 >= *(thresholds++))<<0);
        thresholds = start_th;
        tmp2 = ( (accum1_ts1 >= *(thresholds++)) <<3) +
              ((accum2_ts1 >= *(thresholds++))<<2) +
              ((accum3_ts1 >= *(thresholds++))<<1) +
              ((accum4_ts1 >= *(thresholds++))<<0);*/

        /*uint32_t tmp, tmp2;
        int16_t thresholds1 = *(thresholds++);
        int16_t thresholds2 = *(thresholds++);
        int16_t thresholds3 = *(thresholds++);
        int16_t thresholds4 = *(thresholds++);
        asm volatile("":::"memory");

        tmp =  ((accum1_ts0 >= thresholds1) <<3) +
               ((accum2_ts0 >= thresholds2)<<2) +
               ((accum3_ts0 >= thresholds3)<<1) +
               ((accum4_ts0 >= thresholds4)<<0);
        tmp2 = ( (accum1_ts1 >= thresholds1) <<3) +
               ((accum2_ts1 >= thresholds2)<<2) +
               ((accum3_ts1 >= thresholds3)<<1) +
               ((accum4_ts1 >= thresholds4)<<0);*/

        z |= tmp << (32 - savecounter -4);
        z2 |= tmp2 << (32 - savecounter2-4);

        weights_overstep1  = MODULO_32(weights_overstep4  + dim_ker);

        A1 = A4_next;
        A1 <<= weights_overstep1;
        int next_weights_in_next_word = ((weights_overstep1 + words_leftover_kernel) >=32) | next_pA;
        pA1_next =pA4_next + next_weights_in_next_word;
        A1_next = *(pA1_next);
        weights_mask1 = (1<< (weights_overstep1)) - 1;
        A1 = A1 |  (A1_next >> (32-weights_overstep1) & weights_mask1);

        pA1 = pA4_next;

        savecounter +=4;
        if((savecounter)%32==0) {
            *(pOut++)|=(z>>output_offset);
            savecounter=0;
            z = 0;
        }

        savecounter2 += 4;
        if((savecounter2)%32==0) {
            *(pOut2++)|=(z2>>output_offset2);
            savecounter2=0;
            z2 = 0;
        }

    }



    while(leftover_channels) {
#ifdef DEBUG
            printf("-------------------------\n");
#endif

        pB1 = pInBuffer;
        pB2 = pInBuffer  + ((input_offset + ch_in)>>5);
        B1 = (*pB1);
        B2 = (*pB2);
        B1_next = *(++pB1);
        B2_next = *(++pB2);

        uint32_t input_mask = (1<< input_offset) - 1;
        uint32_t input_mask2 = (1<< input_offset2) - 1;

        B1 <<= input_offset;
        B1=  B1| (B1_next >> (32  - input_offset) & input_mask);
        B2 <<= input_offset2;
        B2= B2| (B2_next >> (32  - input_offset2) & input_mask2);

        int next_weights_in_next_word1 = ((weights_overstep1 + words_leftover_kernel) >=32);

        //Store the conv output
        int accum1_ts0=0;
        int accum1_ts1=0;

        for (int words_in = 0; words_in < words_kernel; words_in++) {

            accum1_ts0 += POPCOUNT(XNOR(B1,A1));
            accum1_ts1 += POPCOUNT(XNOR(B2,A1));

            #ifdef DEBUG
                printf("A:%x\n", A1);
                printf("B:%x\n",B1);
                printf("B2:%x\n",B2);
            #endif

            //Swap values
            B1 = B1_next;
            B2 = B2_next;

            A1 = A1_next;

            B1_next = *(++pB1);
            B2_next = *(++pB2);
            //Next words are already loaded

            B1 <<= input_offset;
            B1= B1 | ( (B1_next >> (32  - input_offset)) & input_mask);

            B2 <<= input_offset2;
            B2= (B2) | ( (B2_next >> (32  - input_offset2)) & input_mask2);

            A1 <<= weights_overstep1;

            int increment_flag = ( words_in != (words_kernel - 1)); //1 if it's not the last iteration
            pA1_next += next_weights_in_next_word1| increment_flag;
            A1_next = *(pA1_next);
            A1 = A1 | ( (A1_next >> (32-weights_overstep1 )) & weights_mask1);

        }



#ifdef DEBUG
            printf("A:%x\n", A1);
            printf("B:%x\n",B1);
            printf("B2:%x\n",B2);
#endif
        accum1_ts0 += POPCOUNT(XNOR(B1,A1) & leftover_mask);
        accum1_ts1 += POPCOUNT(XNOR(B2,A1) & leftover_mask);

#ifdef DEBUG
            printf("ACCUM1_TS0:%d\n", accum1_ts0);
            printf("ACCUM1_TS1:%d\n", accum1_ts1);
#endif

        uint32_t tmp, tmp2;
        tmp = accum1_ts0 >= *(thresholds);
        tmp2 = accum1_ts1 >= *(thresholds++);
        z |= tmp << (32 - savecounter);
        z2 |= tmp2 << (32 - savecounter2);

        weights_overstep1 = MODULO_32(weights_overstep1 + dim_ker);

        //Next set of weights
        A1 = A1_next;
        A1 <<= weights_overstep1;
        int next_weights_in_next_word = ((weights_overstep1 + words_leftover_kernel) >=32) | next_pA;
        pA1_next += next_weights_in_next_word;//((weights_overstep1 + dim_ker) >=32) | (next_pA);
        A1_next = *(pA1_next);
        weights_mask1 = (1<< (weights_overstep1)) - 1;
        A1 = A1 |  (A1_next >> (32-weights_overstep1) & weights_mask1);


        //TODO Find a smarter way to do this
        savecounter +=1;
        if((savecounter)%32==0) {
            *(pOut++)|=(z>>output_offset);
            savecounter=0;
            z = 0;
        }

        savecounter2 += 1;
        if((savecounter2)%32==0) {
            *(pOut2++)|=(z2>>output_offset2);
            savecounter2=0;
            z2 = 0;
        }

        leftover_channels--;
    }

    if(z!=0)
    {
        *(pOut) |= (z>>output_offset);
        //pOut += ((output_offset+ch_out)==32);
    }

    if(z2!=0)
    {
        *(pOut2) |= (z2>>output_offset2);
        pOut2 += ((output_offset2+ch_out)==32);
    }


    return pOut2;


}
