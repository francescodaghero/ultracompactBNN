//#include "stdio.h"
#include "rt/rt_api.h"
#include "../include/kernels.h"

//MACROS
#define DIVIDE_32(x) x>>5
#define DIVIDE_4(x) x>>2

#define MODULO_32(x) (x & 31)
#define MODULO_4(x) (x & 3)


//#define DEBUG 1

//typedef int8_t v4s __attribute__ ((vector_size (4)));
#define LOAD_VECTOR(x) (* ( (v4s*) x  ))
#define SumDotp(a, b, c)        __builtin_pulp_sdotsp4(a, b, c)
#define NEG(x) __builtin_pulp_neg4(x)


void __attribute__((always_inline)) matmul4x2_pooling(
        const int8_t * pInBuffer,
        const uint16_t  dim_in,
        const uint16_t  ch_in,
        const int8_t *  pWeight,
        const uint16_t  ch_out,
        const uint16_t  kernel_size,
        const uint16_t  stride,
        uint32_t *       pOut1,
        uint32_t output_offset1,
        const uint16_t  dim_out,
        const int16_t *thresholds,
        uint8_t pooling_elements_left,
        uint8_t pooling_window,
        uint8_t pooling_stride
) {
    int8_t *pB1;
    int8_t *pB2;
    int8_t *pA1 = pWeight;

    v4s vecA1;
    v4s vecA2;
    v4s vecA3;
    v4s vecA4;
    v4s vecB1;
    v4s vecB2;

    uint32_t *pOut2;


    int dim_ker=ch_in*kernel_size;

    int leftover;


    uint8_t pooling_dependant_increment2=(((-((++pooling_elements_left) > pooling_window) ^ ch_out) & ch_out ) ^ ch_out);
    uint32_t output_offset2=output_offset1 + pooling_dependant_increment2;
    output_offset2=MODULO_32(output_offset2);

    pOut2 = pOut1 + (DIVIDE_32((output_offset1 + pooling_dependant_increment2)));


    for (int out_channel = 0; out_channel < (ch_out)>>2; out_channel++) {
#ifdef DEBUG
        printf("------------------------------------\n");
#endif

        pB1 = pInBuffer;
        pB2 = pB1 + stride*ch_in;

        int8_t *pA2 = (pA1 + dim_ker);
        int8_t *pA3 = (pA2 + dim_ker);
        int8_t *pA4 = (pA3 + dim_ker);

        //Store the conv output
        int accum1_ts1 =0;
        int accum2_ts1 =0;
        int accum3_ts1 =0;
        int accum4_ts1 =0;
        int accum1_ts2 =0;
        int accum2_ts2 =0;
        int accum3_ts2 =0;
        int accum4_ts2 =0;



        for (int j = 0; j < DIVIDE_4(dim_ker); j++) {

            vecB1  = LOAD_VECTOR(pB1);
            vecB2 = LOAD_VECTOR(pB2);

            vecA1 = LOAD_VECTOR(pA1);
            vecA2 = LOAD_VECTOR(pA2);
            vecA3 = LOAD_VECTOR(pA3);
            vecA4 = LOAD_VECTOR(pA4);

            accum1_ts1 =  SumDotp (vecB1,  vecA1,  accum1_ts1);
            accum2_ts1 =  SumDotp (vecB1,  vecA2, accum2_ts1 );
            accum3_ts1 =  SumDotp (vecB1,  vecA3, accum3_ts1 );
            accum4_ts1 =  SumDotp (vecB1,  vecA4, accum4_ts1 );

            accum1_ts2 =  SumDotp (vecB2,  vecA1,  accum1_ts2);
            accum2_ts2 =  SumDotp (vecB2,  vecA2, accum2_ts2 );
            accum3_ts2 =  SumDotp (vecB2,  vecA3, accum3_ts2 );
            accum4_ts2 =  SumDotp (vecB2,  vecA4, accum4_ts2 );

//            asm volatile("": : :"memory");


            pB1+=4;
            pB2+=4;

            pA1  +=4;
            pA2 += 4;
            pA3 += 4;
            pA4 += 4;

#ifdef DEBUG
            printf("ACCUM1_TS1: %d\n", accum1_ts1);
            printf("ACCUM2_TS1: %d\n", accum2_ts1);
            printf("ACCUM3_TS1: %d\n", accum3_ts1);
            printf("ACCUM4_TS1: %d\n", accum4_ts1);
            printf("ACCUM1_TS2: %d\n", accum1_ts2);
            printf("ACCUM2_TS2: %d\n", accum2_ts2);
            printf("ACCUM3_TS2: %d\n", accum3_ts2);
            printf("ACCUM4_TS2: %d\n", accum4_ts2);
#endif


        }

        leftover = MODULO_4(dim_ker);
        while(leftover)
        {

            int8_t B1 = *(pB1++);
            int8_t B2 = *(pB2++);
            int8_t A1 = *(pA1++);
            int8_t A2 = *(pA2++);
            int8_t A3 = *(pA3++);
            int8_t A4 = *(pA4++);
            asm volatile("": : :"memory");


            accum1_ts1 +=  B1*A1;
            accum2_ts1 +=  B1*A2;
            accum3_ts1 +=  B1*A3;
            accum4_ts1 +=  B1*A4;

            accum1_ts2 +=  B2*A1;
            accum2_ts2 +=  B2*A2;
            accum3_ts2 +=  B2*A3;
            accum4_ts2 +=  B2*A4;


            leftover--;
        }


#ifdef DEBUG
        printf("ACCUM1_TS1: %d\n", accum1_ts1);
        printf("ACCUM2_TS1: %d\n", accum2_ts1);
        printf("ACCUM3_TS1: %d\n", accum3_ts1);
        printf("ACCUM4_TS1: %d\n", accum4_ts1);
        printf("ACCUM1_TS2: %d\n", accum1_ts2);
        printf("ACCUM2_TS2: %d\n", accum2_ts2);
        printf("ACCUM3_TS2: %d\n", accum3_ts2);
        printf("ACCUM4_TS2: %d\n", accum4_ts2);
#endif


        //Thresholding and binarization
        uint32_t tmp_results1, tmp_results2;

        tmp_results1  = (accum1_ts1>= *(thresholds))<<3;
        tmp_results2  = (accum1_ts2>= *(thresholds++))<<3;


        tmp_results1 |= (accum2_ts1>= *(thresholds))<<2;
        tmp_results2 |= (accum2_ts2>= *(thresholds++))<<2;

        tmp_results1 |= (accum3_ts1>= *(thresholds))<<1;
        tmp_results2 |= (accum3_ts2>= *(thresholds++))<<1;

        tmp_results1 |= (accum4_ts1>= *(thresholds))<<0;
        tmp_results2 |= (accum4_ts2>= *(thresholds++))<<0;

        uint8_t channel_increment= 4;

        //Save output
        *pOut1 |= tmp_results1<< (32 - output_offset1 - 4);
        output_offset1+=channel_increment;
        pOut1+=DIVIDE_32(output_offset1);
        output_offset1 = MODULO_32(output_offset1);

        //Save output

        *pOut2 |= tmp_results2<< (32 - output_offset2 - 4);
        output_offset2+=channel_increment;
        pOut2+=DIVIDE_32(output_offset2);
        output_offset2 = MODULO_32(output_offset2);

        pA1 = pA4;

    }

    int channel_left = MODULO_4(ch_out);

    //LEFTOVER CHANNELS
    while(channel_left) {
#ifdef DEBUG
        printf("------------------------------------\n");
#endif

        pB1 = pInBuffer;
        pB2 = pB1 + stride*ch_in;



        //Store the conv output
        int accum1_ts1 =0;
        int accum1_ts2 =0;

        for (int j = 0; j < DIVIDE_4(dim_ker); j++) {

            vecB1  = LOAD_VECTOR(pB1);
            vecB2 = LOAD_VECTOR(pB2);

            vecA1 = LOAD_VECTOR(pA1);

            accum1_ts1 =  SumDotp (vecB1,  vecA1,  accum1_ts1);

            accum1_ts2 =  SumDotp (vecB2,  vecA1,  accum1_ts2);


            pB1+=4;
            pB2+=4;

            pA1  += 4;

#ifdef DEBUG
            printf("ACCUM1_TS1: %d\n", accum1_ts1);
            printf("ACCUM2_TS1: %d\n", accum2_ts1);
            printf("ACCUM3_TS1: %d\n", accum3_ts1);
            printf("ACCUM4_TS1: %d\n", accum4_ts1);
            printf("ACCUM1_TS2: %d\n", accum1_ts2);
            printf("ACCUM2_TS2: %d\n", accum2_ts2);
            printf("ACCUM3_TS2: %d\n", accum3_ts2);
            printf("ACCUM4_TS2: %d\n", accum4_ts2);
#endif


        }

        leftover = MODULO_4(dim_ker);
        while(leftover)
        {
            int8_t B1 = *(pB1++);
            int8_t B2 = *(pB2++);
            int8_t A1 = *(pA1++);
            asm volatile("":::"memory");
            accum1_ts1 +=  B1*A1;
            accum1_ts2 +=  B2*A1;

            leftover--;
        }

#ifdef DEBUG
        printf("ACCUM1_TS1: %d\n", accum1_ts1);
        printf("ACCUM2_TS1: %d\n", accum2_ts1);
        printf("ACCUM3_TS1: %d\n", accum3_ts1);
        printf("ACCUM4_TS1: %d\n", accum4_ts1);
        printf("ACCUM1_TS2: %d\n", accum1_ts2);
        printf("ACCUM2_TS2: %d\n", accum2_ts2);
        printf("ACCUM3_TS2: %d\n", accum3_ts2);
        printf("ACCUM4_TS2: %d\n", accum4_ts2);
#endif
        //Thresholding and binarization
        uint32_t tmp_results1, tmp_results2;
//        printf("%d %d\n", accum1_ts1, accum1_ts2);


        tmp_results1  = (accum1_ts1>= *(thresholds));
        tmp_results2  = (accum1_ts2>= *(thresholds++));


        //Save output
        *pOut1 |= tmp_results1<< (32 - output_offset1 - 1);
        output_offset1+=1;
        pOut1+=DIVIDE_32(output_offset1);
        output_offset1 = MODULO_32(output_offset1);

        //Save output
        *pOut2 |= tmp_results2<< (32 - output_offset2 - 1);
        output_offset2+=1;
        pOut2+=DIVIDE_32(output_offset2);
        output_offset2 = MODULO_32(output_offset2);

        channel_left--;

    }
}
