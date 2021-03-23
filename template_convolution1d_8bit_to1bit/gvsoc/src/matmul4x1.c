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


uint32_t* __attribute__((always_inline)) matmul4x1(
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
        const int16_t *thresholds
) {
    int8_t *pB1;
    int8_t *pA1=pWeight;

    v4s vecA1;
    v4s vecA2;
    v4s vecA3;
    v4s vecA4;
    v4s vecB1;

    int dim_ker=ch_in*kernel_size;

    int leftover;


    //Variabile per l'output
    uint32_t z1 = 0;
    uint32_t savecounter1 = output_offset1;



    for (int out_channel = 0; out_channel < (ch_out)>>2; out_channel++) {
#ifdef DEBUG
        printf("------------------------------------\n");
#endif

        pB1 = pInBuffer;

        int8_t *pA2 = (pA1 + dim_ker);
        int8_t *pA3 = (pA2 + dim_ker);
        int8_t *pA4 = (pA3 + dim_ker);

        //Store the conv output
        int accum1_ts1 =0;
        int accum2_ts1 =0;
        int accum3_ts1 =0;
        int accum4_ts1 =0;



        for (int j = 0; j < DIVIDE_4(dim_ker); j++) {

            vecB1  = LOAD_VECTOR(pB1);

            vecA1 = LOAD_VECTOR(pA1);
            vecA2 = LOAD_VECTOR(pA2);
            vecA3 = LOAD_VECTOR(pA3);
            vecA4 = LOAD_VECTOR(pA4);

            accum1_ts1 =  SumDotp (vecB1,  vecA1,  accum1_ts1);
            accum2_ts1 =  SumDotp (vecB1,  vecA2, accum2_ts1 );
            accum3_ts1 =  SumDotp (vecB1,  vecA3, accum3_ts1 );
            accum4_ts1 =  SumDotp (vecB1,  vecA4, accum4_ts1 );


            pB1+=4;

            pA1  += 4;
            pA2 += 4;
            pA3 += 4;
            pA4 += 4;

#ifdef DEBUG
            printf("ACCUM1_TS1: %d\n", accum1_ts1);
            printf("ACCUM2_TS1: %d\n", accum2_ts1);
            printf("ACCUM3_TS1: %d\n", accum3_ts1);
            printf("ACCUM4_TS1: %d\n", accum4_ts1);
#endif


        }

        leftover = MODULO_4(dim_ker);
        while(leftover)
        {
            int8_t B1 = *(pB1++);
            int8_t A1 = *(pA1++);
            int8_t A2 = *(pA2++);
            int8_t A3 = *(pA3++);
            int8_t A4 = *(pA4++);

            accum1_ts1 +=  B1*A1;
            accum2_ts1 +=  B1*A2;
            accum3_ts1 +=  B1*A3;
            accum4_ts1 +=  B1*A4;

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
        tmp_results1 |= (accum2_ts1>= *(thresholds))<<2;
        tmp_results1 |= (accum3_ts1>= *(thresholds))<<1;
        tmp_results1 |= (accum4_ts1>= *(thresholds))<<0;


        //Save output
        *pOut1 |= tmp_results1<< (32 - output_offset1 - 4);
        output_offset1+=4;
        pOut1+=DIVIDE_32(output_offset1);
        output_offset1 = MODULO_32(output_offset1);

        pA1=pA4;

    }

    int channel_left = MODULO_4(ch_out);

    //LEFTOVER CHANNELS
    while(channel_left--) {
#ifdef DEBUG
        printf("------------------------------------\n");
#endif

        pB1 = pInBuffer;

        //Store the conv output
        int accum1_ts1 =0;

        for (int j = 0; j < DIVIDE_4(dim_ker); j++) {

            vecB1  = LOAD_VECTOR(pB1);
            vecA1 = LOAD_VECTOR(pA1);

            accum1_ts1 =  SumDotp (vecB1,  vecA1,  accum1_ts1);

            pB1+=4;
            pA1  += 4;

#ifdef DEBUG
            printf("ACCUM1_TS1: %d\n", accum1_ts1);
            printf("ACCUM1_TS2: %d\n", accum1_ts2);
#endif


        }

        leftover = MODULO_4(dim_ker);
        while(leftover)
        {
            int8_t B1 = *(pB1++);
            int8_t A1 = *(pA1++);
            asm volatile("":::"memory");
            accum1_ts1 +=  B1*A1;

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

        tmp_results1  = (accum1_ts1>= *(thresholds++));


        //Save output
        *pOut1 |= tmp_results1<< (32 - output_offset1 - 1);
        output_offset1+=1;
        pOut1+=DIVIDE_32(output_offset1);
        output_offset1 = MODULO_32(output_offset1);
        channel_left--;

    }



    return pOut1;
}