#include "rt/rt_api.h"
#include "../include/kernels.h"

//MACROS
#define DIVIDE_32(x) x>>5
#define MODULO_32(x) (x & 0x1F)
#define XNOR(B,A) (~(B ^ A))

#define POPCOUNT(x) __builtin_pulp_cnt(x)

//#define DEBUG


uint32_t *xnorpop4x4(
        const uint32_t * pInBuffer,
        const uint32_t input_offset1,
        const int  dim_ker,//
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
    uint32_t input_offset3;
    uint32_t input_offset4;
    uint32_t *pA1, *pA1_next;
    uint32_t A1;

    uint32_t *pOut2;
    uint32_t *pOut3;
    uint32_t *pOut4;

    //Variables
    //NxM dependant
    //N
    uint32_t leftover_channels = ch_out %4;
    //M
    input_offset2 = MODULO_32(input_offset1 + ch_in);
    input_offset3 = MODULO_32(input_offset2 + ch_in);
    input_offset4 = MODULO_32(input_offset3 + ch_in);
    //Other variables
    int words_kernel = DIVIDE_32(dim_ker);
    const uint32_t words_leftover_kernel = MODULO_32(dim_ker);
    uint32_t leftover_mask = words_leftover_kernel!=0 ? ~((1 << (32 - words_leftover_kernel )) - 1) : 0x0;

    uint32_t weights_overstep1 =0;
    uint32_t weights_mask1 = (1<< (weights_overstep1)) - 1;


    uint32_t output_offset2 = MODULO_32(output_offset1 + ch_out);
    uint32_t output_offset3 = MODULO_32(output_offset2 + ch_out);
    uint32_t output_offset4 = MODULO_32(output_offset3 + ch_out);

    pA1 = pWeight;
    pA1_next = pWeight + (dim_ker >= 32);
    pOut2 = pOut1 + (DIVIDE_32((output_offset1 + ch_out)));
    pOut3 = pOut2 + (DIVIDE_32((output_offset2 + ch_out)));
    pOut4 = pOut3 + (DIVIDE_32((output_offset3 + ch_out)));

    //Load
    A1 = *(pA1);

    if((ch_out)>>2) {
        int out_channel = 0;
        do {
            #ifdef DEBUG
        printf("-------------------------\n");
        #endif
        uint32_t B2;
        uint32_t B3;
        uint32_t B4;
        //Update pointers and variables
        uint32_t *pB1 = pInBuffer;
        uint32_t *pB2 = pB1  + (DIVIDE_32((input_offset1 + ch_in)));
        uint32_t *pB3 = pB2  + (DIVIDE_32((input_offset2 + ch_in)));
        uint32_t *pB4 = pB3  + (DIVIDE_32((input_offset3 + ch_in)));

        uint32_t input_mask1 = (1<< input_offset1) - 1;
        uint32_t input_mask2 = (1<< input_offset2) - 1;
        uint32_t input_mask3 = (1<< input_offset3) - 1;
        uint32_t input_mask4 = (1<< input_offset4) - 1;

        //Additional weights
        uint32_t weights_overstep2 = MODULO_32(weights_overstep1)+words_leftover_kernel;
        uint32_t weights_overstep3 = MODULO_32(weights_overstep2)+words_leftover_kernel;
        uint32_t weights_overstep4 = MODULO_32(weights_overstep3)+words_leftover_kernel;

        uint32_t *pA2 = pA1 + words_kernel + (weights_overstep2 >=32); //Point pAI all'inizio dei pesi
        uint32_t *pA3 = pA2 + words_kernel + (weights_overstep3 >=32); //Point pAI all'inizio dei pesi
        uint32_t *pA4 = pA3 + words_kernel + (weights_overstep4 >=32); //Point pAI all'inizio dei pesi

        weights_overstep2 = MODULO_32(weights_overstep2);
        weights_overstep3 = MODULO_32(weights_overstep3);
        weights_overstep4 = MODULO_32(weights_overstep4);



        uint32_t *pA2_next = pA2 + (((weights_overstep2 + words_leftover_kernel) >= 32) | dim_ker >= 32);
        uint32_t *pA3_next = pA3 + (((weights_overstep3 + words_leftover_kernel) >= 32) | dim_ker >= 32);
        uint32_t *pA4_next = pA4 + (((weights_overstep4 + words_leftover_kernel) >= 32) | dim_ker >= 32);

        uint32_t weights_mask2 = (1<< (weights_overstep2)) - 1;
        uint32_t weights_mask3 = (1<< (weights_overstep3)) - 1;
        uint32_t weights_mask4 = (1<< (weights_overstep4)) - 1;

        //Store the conv output
        int accum1_ts1 =0;
        int accum1_ts2 =0;
        int accum1_ts3 =0;
        int accum1_ts4 =0;
        int accum2_ts1 =0;
        int accum2_ts2 =0;
        int accum2_ts3 =0;
        int accum2_ts4 =0;
        int accum3_ts1 =0;
        int accum3_ts2 =0;
        int accum3_ts3 =0;
        int accum3_ts4 =0;
        int accum4_ts1 =0;
        int accum4_ts2 =0;
        int accum4_ts3 =0;
        int accum4_ts4 =0;

        //LOAD

        uint32_t A2 = *(pA2);
        uint32_t A3 = *(pA3);
        uint32_t A4 = *(pA4);

        //Operations on loaded values
        B1= ((*(pB1++)) << input_offset1)| (*(pB1) >> (32  - input_offset1) & input_mask1);
        B2= ((*(pB2++)) << input_offset2)| (*(pB2) >> (32  - input_offset2) & input_mask2);
        B3= ((*(pB3++)) << input_offset3)| (*(pB3) >> (32  - input_offset3) & input_mask3);
        B4= ((*(pB4++)) << input_offset4)| (*(pB4) >> (32  - input_offset4) & input_mask4);


        A2 = A2 << weights_overstep2 |  (*(pA2_next) >> (32-weights_overstep2) & weights_mask2);
        A3 = A3 << weights_overstep3 |  (*(pA3_next) >> (32-weights_overstep3) & weights_mask3);
        A4 = A4 << weights_overstep4 |  (*(pA4_next) >> (32-weights_overstep4) & weights_mask4);

        if((words_kernel)>0) {
            for (int words_in = 0; words_in < (words_kernel-1); words_in++) {

                accum1_ts1 += POPCOUNT(XNOR(B1,A1));
                accum1_ts2 += POPCOUNT(XNOR(B2,A1));
                accum1_ts3 += POPCOUNT(XNOR(B3,A1));
                accum1_ts4 += POPCOUNT(XNOR(B4,A1));
                accum2_ts1 += POPCOUNT(XNOR(B1,A2));
                accum2_ts2 += POPCOUNT(XNOR(B2,A2));
                accum2_ts3 += POPCOUNT(XNOR(B3,A2));
                accum2_ts4 += POPCOUNT(XNOR(B4,A2));
                accum3_ts1 += POPCOUNT(XNOR(B1,A3));
                accum3_ts2 += POPCOUNT(XNOR(B2,A3));
                accum3_ts3 += POPCOUNT(XNOR(B3,A3));
                accum3_ts4 += POPCOUNT(XNOR(B4,A3));
                accum4_ts1 += POPCOUNT(XNOR(B1,A4));
                accum4_ts2 += POPCOUNT(XNOR(B2,A4));
                accum4_ts3 += POPCOUNT(XNOR(B3,A4));
                accum4_ts4 += POPCOUNT(XNOR(B4,A4));



                #ifdef DEBUG
                printf("A1:%x\n",A1);
                printf("A2:%x\n",A2);
                printf("A3:%x\n",A3);
                printf("A4:%x\n",A4);
                printf("B1:%x\n",B1);
                printf("B2:%x\n",B2);
                printf("B3:%x\n",B3);
                printf("B4:%x\n",B4);
                #endif

                //Swap values
                A1 = *(pA1_next);
                A2 = *(pA2_next);
                A3 = *(pA3_next);
                A4 = *(pA4_next);


                B1= ((*(pB1++)) << input_offset1) | ( (*(pB1) >> (32  - input_offset1)) & input_mask1);
                B2= ((*(pB2++)) << input_offset2) | ( (*(pB2) >> (32  - input_offset2)) & input_mask2);
                B3= ((*(pB3++)) << input_offset3) | ( (*(pB3) >> (32  - input_offset3)) & input_mask3);
                B4= ((*(pB4++)) << input_offset4) | ( (*(pB4) >> (32  - input_offset4)) & input_mask4);
                A1 <<= weights_overstep1;
                A2 <<= weights_overstep2;
                A3 <<= weights_overstep3;
                A4 <<= weights_overstep4;

                pA1_next+=1;
                pA2_next+=1;
                pA3_next+=1;
                pA4_next+=1;

                A1 = A1 | ( (*pA1_next >> (32-weights_overstep1 )) & weights_mask1);
                A2 = A2 | ( (*pA2_next >> (32-weights_overstep2 )) & weights_mask2);
                A3 = A3 | ( (*pA3_next >> (32-weights_overstep3 )) & weights_mask3);
                A4 = A4 | ( (*pA4_next >> (32-weights_overstep4 )) & weights_mask4);

            }


            accum1_ts1 += POPCOUNT(XNOR(B1,A1));
            accum1_ts2 += POPCOUNT(XNOR(B2,A1));
            accum1_ts3 += POPCOUNT(XNOR(B3,A1));
            accum1_ts4 += POPCOUNT(XNOR(B4,A1));
            accum2_ts1 += POPCOUNT(XNOR(B1,A2));
            accum2_ts2 += POPCOUNT(XNOR(B2,A2));
            accum2_ts3 += POPCOUNT(XNOR(B3,A2));
            accum2_ts4 += POPCOUNT(XNOR(B4,A2));
            accum3_ts1 += POPCOUNT(XNOR(B1,A3));
            accum3_ts2 += POPCOUNT(XNOR(B2,A3));
            accum3_ts3 += POPCOUNT(XNOR(B3,A3));
            accum3_ts4 += POPCOUNT(XNOR(B4,A3));
            accum4_ts1 += POPCOUNT(XNOR(B1,A4));
            accum4_ts2 += POPCOUNT(XNOR(B2,A4));
            accum4_ts3 += POPCOUNT(XNOR(B3,A4));
            accum4_ts4 += POPCOUNT(XNOR(B4,A4));
            //asm volatile("":::"memory");

            #ifdef DEBUG
                            printf("A1:%x\n",A1);
                            printf("A2:%x\n",A2);
                            printf("A3:%x\n",A3);
                            printf("A4:%x\n",A4);
                            printf("B1:%x\n",B1);
                            printf("B2:%x\n",B2);
                            printf("B3:%x\n",B3);
                            printf("B4:%x\n",B4);
            #endif

            //Swap values
            B1 = *(pB1++);
            B2 = *(pB2++);
            B3 = *(pB3++);
            B4 = *(pB4++);
            A1 = *(pA1_next);
            A2 = *(pA2_next);
            A3 = *(pA3_next);
            A4 = *(pA4_next);


            B1= (B1 << input_offset1) | ( (*(pB1) >> (32  - input_offset1)) & input_mask1);
            B2= (B2 << input_offset2) | ( (*(pB2) >> (32  - input_offset2)) & input_mask2);
            B3= (B3 << input_offset3) | ( (*(pB3) >> (32  - input_offset3)) & input_mask3);
            B4= (B4 << input_offset4) | ( (*(pB4) >> (32  - input_offset4)) & input_mask4);
            A1 <<= weights_overstep1;
            A2 <<= weights_overstep2;
            A3 <<= weights_overstep3;
            A4 <<= weights_overstep4;

            pA1_next += ((weights_overstep1 + words_leftover_kernel) >= 32);
            pA2_next += ((weights_overstep2 + words_leftover_kernel) >= 32);
            pA3_next += ((weights_overstep3 + words_leftover_kernel) >= 32);
            pA4_next += ((weights_overstep4 + words_leftover_kernel) >= 32);


            A1 = A1 | ( (*pA1_next >> (32-weights_overstep1 )) & weights_mask1);
            A2 = A2 | ( (*pA2_next >> (32-weights_overstep2 )) & weights_mask2);
            A3 = A3 | ( (*pA3_next >> (32-weights_overstep3 )) & weights_mask3);
            A4 = A4 | ( (*pA4_next >> (32-weights_overstep4 )) & weights_mask4);

        }

        //ODD KERNELS or DIM_KER < 32
        #ifdef DEBUG
        printf("A1:%x\n", A1 & leftover_mask);
        printf("A2:%x\n", A2 & leftover_mask);
        printf("A3:%x\n", A3 & leftover_mask);
        printf("A4:%x\n", A4 & leftover_mask);
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
        accum3_ts1 += POPCOUNT(XNOR(B1,A3) & leftover_mask);
        accum3_ts2 += POPCOUNT(XNOR(B2,A3) & leftover_mask);
        accum3_ts3 += POPCOUNT(XNOR(B3,A3) & leftover_mask);
        accum3_ts4 += POPCOUNT(XNOR(B4,A3) & leftover_mask);
        accum4_ts1 += POPCOUNT(XNOR(B1,A4) & leftover_mask);
        accum4_ts2 += POPCOUNT(XNOR(B2,A4) & leftover_mask);
        accum4_ts3 += POPCOUNT(XNOR(B3,A4) & leftover_mask);
        accum4_ts4 += POPCOUNT(XNOR(B4,A4) & leftover_mask);


        #ifdef DEBUG
        printf("ACCUM1 TS 1:%d\n", accum1_ts1);
        printf("ACCUM1 TS 2:%d\n", accum1_ts2);
        printf("ACCUM1 TS 3:%d\n", accum1_ts3);
        printf("ACCUM1 TS 4:%d\n", accum1_ts4);
        printf("ACCUM2 TS 1:%d\n", accum2_ts1);
        printf("ACCUM2 TS 2:%d\n", accum2_ts2);
        printf("ACCUM2 TS 3:%d\n", accum2_ts3);
        printf("ACCUM2 TS 4:%d\n", accum2_ts4);
        printf("ACCUM3 TS 1:%d\n", accum3_ts1);
        printf("ACCUM3 TS 2:%d\n", accum3_ts2);
        printf("ACCUM3 TS 3:%d\n", accum3_ts3);
        printf("ACCUM3 TS 4:%d\n", accum3_ts4);
        printf("ACCUM4 TS 1:%d\n", accum4_ts1);
        printf("ACCUM4 TS 2:%d\n", accum4_ts2);
        printf("ACCUM4 TS 3:%d\n", accum4_ts3);
        printf("ACCUM4 TS 4:%d\n", accum4_ts4);

        #endif

        //THRESHOLDING AND BINARIZATION
        *pOut1 |= (accum1_ts1 >= *(thresholds)) << (32 - output_offset1 - 1);
        *pOut2 |= (accum1_ts2 >= *(thresholds)) << (32 - output_offset2 - 1);
        *pOut3 |= (accum1_ts3 >= *(thresholds)) << (32 - output_offset3 - 1);
        *pOut4 |= (accum1_ts4 >= *(thresholds++)) << (32 - output_offset4 -1);

        *pOut1 |= (accum2_ts1 >= *(thresholds)) << (32 - output_offset1 - 2 );
        *pOut2 |= (accum2_ts2 >= *(thresholds)) << (32 - output_offset2 - 2 );
        *pOut3 |= (accum2_ts3 >= *(thresholds)) << (32 - output_offset3 - 2 );
        *pOut4 |= (accum2_ts4 >= *(thresholds++)) << (32 - output_offset4 - 2 );
        *pOut1 |= (accum3_ts1 >= *(thresholds)) << (32 - output_offset1 - 3 );
        *pOut2 |= (accum3_ts2 >= *(thresholds)) << (32 - output_offset2 - 3 );
        *pOut3 |= (accum3_ts3 >= *(thresholds)) << (32 - output_offset3 - 3 );
        *pOut4 |= (accum3_ts4 >= *(thresholds++)) << (32 - output_offset4 - 3 );
        *pOut1 |= (accum4_ts1 >= *(thresholds)) << (32 - output_offset1 - 4 );
        *pOut2 |= (accum4_ts2 >= *(thresholds)) << (32 - output_offset2 - 4 );
        *pOut3 |= (accum4_ts3 >= *(thresholds)) << (32 - output_offset3 - 4 );
        *pOut4 |= (accum4_ts4 >= *(thresholds++)) << (32 - output_offset4 - 4 );

        weights_overstep1  = MODULO_32(weights_overstep4   + dim_ker);

        A1 = *(pA4_next);
        pA1_next =pA4_next + ( ((weights_overstep1 + words_leftover_kernel) >= 32) | dim_ker >= 32);
        weights_mask1 = (1<< (weights_overstep1)) - 1;
        A1 = (A1 << weights_overstep1) |  (*(pA1_next) >> (32-weights_overstep1) & weights_mask1);

        pA1 = pA4_next;

        output_offset1+= 4;
        pOut1 += DIVIDE_32(output_offset1);
        output_offset1 =MODULO_32(output_offset1);
        output_offset2+= 4;
        pOut2 += DIVIDE_32(output_offset2);
        output_offset2 =MODULO_32(output_offset2);
        output_offset3+= 4;
        pOut3 += DIVIDE_32(output_offset3);
        output_offset3 =MODULO_32(output_offset3);
        output_offset4+= 4;
        pOut4 += DIVIDE_32(output_offset4);
        output_offset4 =MODULO_32(output_offset4);

            out_channel++;
        } while (out_channel < (ch_out) >> 2);


    }

    while(leftover_channels) {
#ifdef DEBUG
        printf("-------------------------\n");
#endif
        //Update pointers and variables
        uint32_t *pB1 = pInBuffer;
        uint32_t *pB2 = pB1 + (DIVIDE_32((input_offset1 + ch_in)));
        uint32_t *pB3 = pB2 + (DIVIDE_32((input_offset2 + ch_in)));
        uint32_t *pB4 = pB3 + (DIVIDE_32((input_offset3 + ch_in)));

        uint32_t input_mask1 = (1 << input_offset1) - 1;
        uint32_t input_mask2 = (1 << input_offset2) - 1;
        uint32_t input_mask3 = (1 << input_offset3) - 1;
        uint32_t input_mask4 = (1 << input_offset4) - 1;

        //Additional weights

        //Store the conv output
        int accum1_ts1 = 0;
        int accum1_ts2 = 0;
        int accum1_ts3 = 0;
        int accum1_ts4 = 0;

        //LOAD


        //Operations on loaded values
        B1 = ((*(pB1++)) << input_offset1) | (*(pB1) >> (32 - input_offset1) & input_mask1);
        uint32_t B2 = ((*(pB2++)) << input_offset2) | (*(pB2) >> (32 - input_offset2) & input_mask2);
        uint32_t B3 = ((*(pB3++)) << input_offset3) | (*(pB3) >> (32 - input_offset3) & input_mask3);
        uint32_t B4 = ((*(pB4++)) << input_offset4) | (*(pB4) >> (32 - input_offset4) & input_mask4);



        if((words_kernel)>0) {
        int words_in = 0;
        do {
            accum1_ts1 += POPCOUNT(XNOR(B1, A1));
            accum1_ts2 += POPCOUNT(XNOR(B2, A1));
            accum1_ts3 += POPCOUNT(XNOR(B3, A1));
            accum1_ts4 += POPCOUNT(XNOR(B4, A1));
//                asm volatile("":::"memory");

#ifdef DEBUG
            printf("A1:%x\n",A1);
                printf("B1:%x\n",B1);
                printf("B2:%x\n",B2);
                printf("B3:%x\n",B3);
                printf("B4:%x\n",B4);
#endif

            B1 = *(pB1++);
            B2 = *(pB2++);
            B3 = *(pB3++);
            B4 = *(pB4++);
            //Swap values
            A1 = *(pA1_next);


            B1 = ((B1) << input_offset1) | ((*(pB1) >> (32 - input_offset1)) & input_mask1);
            B2 = ((B2) << input_offset2) | ((*(pB2) >> (32 - input_offset2)) & input_mask2);
            B3 = ((B3) << input_offset3) | ((*(pB3) >> (32 - input_offset3)) & input_mask3);
            B4 = ((B4) << input_offset4) | ((*(pB4) >> (32 - input_offset4)) & input_mask4);
            A1 <<= weights_overstep1;
            pA1_next += ((weights_overstep1 + words_leftover_kernel) >= 32) | (words_in != (words_kernel - 1));


            A1 = A1 | ((*pA1_next >> (32 - weights_overstep1)) & weights_mask1);

            words_in++;
        } while (words_in < words_kernel);
    }
//            for (int words_in = 0; words_in < (words_kernel); words_in++) {
//
//
//                accum1_ts1 += POPCOUNT(XNOR(B1,A1));
//                accum1_ts2 += POPCOUNT(XNOR(B2,A1));
//                accum1_ts3 += POPCOUNT(XNOR(B3,A1));
//                accum1_ts4 += POPCOUNT(XNOR(B4,A1));
////                asm volatile("":::"memory");
//
//                #ifdef DEBUG
//                printf("A1:%x\n",A1);
//                printf("B1:%x\n",B1);
//                printf("B2:%x\n",B2);
//                printf("B3:%x\n",B3);
//                printf("B4:%x\n",B4);
//                #endif
//
//                B1 = *(pB1++);
//                B2 = *(pB2++);
//                B3 = *(pB3++);
//                B4 = *(pB4++);
//                //Swap values
//                A1 = *(pA1_next);
//
//
//                B1= ((B1) << input_offset1) | ( (*(pB1) >> (32  - input_offset1)) & input_mask1);
//                B2= ((B2) << input_offset2) | ( (*(pB2) >> (32  - input_offset2)) & input_mask2);
//                B3= ((B3) << input_offset3) | ( (*(pB3) >> (32  - input_offset3)) & input_mask3);
//                B4= ((B4) << input_offset4) | ( (*(pB4) >> (32  - input_offset4)) & input_mask4);
//                A1 <<= weights_overstep1;
//                pA1_next += ((weights_overstep1 + words_leftover_kernel) >= 32) | (words_in !=(words_kernel-1));
//
//
//                A1 = A1 | ( (*pA1_next >> (32-weights_overstep1 )) & weights_mask1);
//
//            }


        //}

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
        *pOut1 |= (accum1_ts1 >= *(thresholds)) << (32 - output_offset1 - 1);
        *pOut2 |= (accum1_ts2 >= *(thresholds)) << (32 - output_offset2 - 1);
        *pOut3 |= (accum1_ts3 >= *(thresholds)) << (32 - output_offset3 - 1);
        *pOut4 |= (accum1_ts4 >= *(thresholds++)) << (32 - output_offset4 -1);

//       asm volatile("":::"memory");

        weights_overstep1  = MODULO_32(weights_overstep1   + dim_ker);

        A1 = *(pA1_next);
        pA1_next =pA1_next + ( ((weights_overstep1 + words_leftover_kernel) >= 32) | dim_ker >= 32);
        weights_mask1 = (1<< (weights_overstep1)) - 1;
        A1 = (A1 << weights_overstep1) |  (*(pA1_next) >> (32-weights_overstep1) & weights_mask1);

        output_offset1+=1;
        pOut1 += DIVIDE_32(output_offset1);
        output_offset1 =MODULO_32(output_offset1);
        output_offset2+= 1;
        pOut2 += DIVIDE_32(output_offset2);
        output_offset2 =MODULO_32(output_offset2);
        output_offset3+= 1;
        pOut3 += DIVIDE_32(output_offset3);
        output_offset3 =MODULO_32(output_offset3);
        output_offset4+= 1;
        pOut4 += DIVIDE_32(output_offset4);
        output_offset4 =MODULO_32(output_offset4);


    leftover_channels--;
    }


    return pOut4;


}