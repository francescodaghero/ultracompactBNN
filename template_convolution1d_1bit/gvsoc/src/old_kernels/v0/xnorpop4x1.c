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

uint32_t *xnorpop4x1(
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
    uint32_t leftover_channels = ch_out %4;

    uint32_t next_pA = dim_ker>=32;
    uint32_t words_kernel = DIVIDE_32(dim_ker);
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


    //Weights variables
    uint32_t *pA2;
    uint32_t *pA2_next;
    uint32_t A2, A2_next;
    uint32_t weights_mask2;
    int accum2;

    //Weights variables
    uint32_t *pA3;
    uint32_t *pA3_next;
    uint32_t A3, A3_next;
    uint32_t weights_mask3;
    int accum3;


    //Weights variables
    uint32_t *pA4;
    uint32_t *pA4_next;
    uint32_t A4, A4_next;
    uint32_t weights_mask4;
    int accum4;


    pA1 = pWeight;
    A1 = (*pA1);
    pA1_next = pA1 + (next_pA);
    A1_next = *(pA1_next);

    uint32_t weights_overstep1 =0;
    uint32_t weights_mask1 = (1<< (weights_overstep1)) - 1;


    //Variabile per l'output
    uint32_t z = 0;
    uint32_t savecounter = output_offset;
    uint32_t out_mask = 0x80000000;

    //Iterazioni sui canali in output
    for(int out_channel=0; out_channel< (ch_out>>2); out_channel++) {

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

        //Additional weights
        uint32_t weights_overstep2 = MODULO_32(weights_overstep1+words_leftover_kernel);
        pA2 = pA1 + words_kernel + ((weights_overstep1+words_leftover_kernel) >=32); //Point pA2 all'inizio dei pesi
        int next_weights_in_next_word2 = ((weights_overstep2 + words_leftover_kernel) >=32) | next_pA;
        pA2_next = pA2 + next_weights_in_next_word2;
        A2 = (*(pA2)) << weights_overstep2;
        A2_next = *(pA2_next);
        weights_mask2 = (1<< (weights_overstep2)) - 1;
        A2 = A2 |  (A2_next >> (32-weights_overstep2) & weights_mask2);

        //Additional weights
        uint32_t weights_overstep3 = MODULO_32(weights_overstep2+words_leftover_kernel);
        pA3 = pA2 + words_kernel + ((weights_overstep2+words_leftover_kernel) >=32); //Point pA2 all'inizio dei pesi
        int next_weights_in_next_word3 = ((weights_overstep3 + words_leftover_kernel) >=32) | next_pA;
        pA3_next = pA3 + next_weights_in_next_word3;
        A3 = (*(pA3)) << weights_overstep3;
        A3_next = *(pA3_next);
        weights_mask3 = (1<< (weights_overstep3)) - 1;
        A3 = A3 |  (A3_next >> (32-weights_overstep3) & weights_mask3);


        //Additional weights
        uint32_t weights_overstep4 = MODULO_32(weights_overstep3+words_leftover_kernel);
        pA4 = pA3 + words_kernel + ((weights_overstep3+words_leftover_kernel) >=32); //Point pA2 all'inizio dei pesi
        int next_weights_in_next_word4 = ((weights_overstep4 + words_leftover_kernel) >=32) | next_pA;
        pA4_next = pA4 + next_weights_in_next_word4;
        A4 = (*(pA4)) << weights_overstep4;
        A4_next = *(pA4_next);
        weights_mask4 = (1<< (weights_overstep4)) - 1;
        A4 = A4 |  (A4_next >> (32-weights_overstep4) & weights_mask4);


        //Store the conv output
        accum1 =0;
        accum2 =0;
        accum3 =0;
        accum4 =0;

        for (int words_in = 0; words_in < words_kernel; words_in++) {

            accum1 += POPCOUNT(XNOR(B1,A1));
            accum2 += POPCOUNT(XNOR(B1,A2));
            accum3 += POPCOUNT(XNOR(B1,A3));
            accum4 += POPCOUNT(XNOR(B1,A4));

            if(DEBUG) {
                printf("A:%x\n", A1);
                printf("A2:%x\n", A2);
                printf("A3:%x\n", A2);
                printf("A4:%x\n", A2);
                printf("B:%x\n",B1);

            }

            //Next words are already loaded
            B1 = B1_next;
            B1 <<= input_offset;
            B1_next = *(++pB1_next);
            B1= (B1) | ( (B1_next >> (32  - input_offset)) & input_mask);


            A1 = A1_next;
            A1 <<= weights_overstep1 ;
            //A2 deve sempre puntare alla parola con i prossimi pesi necessari al seguente calcolo
            uint32_t incr = ((words_leftover_kernel + weights_overstep1 ) >= 32) | ( words_in != (words_kernel - 1));
            pA1_next += incr;
            A1_next = (*(pA1_next));
            A1 = A1 | ( (A1_next >> (32-weights_overstep1 )) & weights_mask1);

            A2 = A2_next;
            A2 <<= weights_overstep2;
            uint32_t incr2 = ((words_leftover_kernel + weights_overstep2) >= 32) | ( words_in != (words_kernel - 1));
            pA2_next += incr2;
            A2_next = (*(pA2_next));
            A2 = A2 | ( (A2_next >> (32-weights_overstep2)) & weights_mask2);

            A3 = A3_next;
            A3 <<= weights_overstep3;
            uint32_t incr3 = ((words_leftover_kernel + weights_overstep3) >= 32) | ( words_in != (words_kernel - 1));
            pA3_next += incr3;
            A3_next = (*(pA3_next));
            A3 = A3 | ( (A3_next >> (32-weights_overstep3)) & weights_mask3);

            A4 = A4_next;
            A4 <<= weights_overstep4;
            uint32_t incr4 = ((words_leftover_kernel + weights_overstep4) >= 32) | ( words_in != (words_kernel - 1));
            pA4_next += incr4;
            A4_next = (*(pA4_next));
            A4 = A4 | ( (A4_next >> (32-weights_overstep4)) & weights_mask4);

        }


        //ODD KERNELS or DIM_KER < 32
        if(DEBUG) {
            printf("A:%x\n", A1 & leftover_mask);
            printf("A2:%x\n", A2 & leftover_mask);
            printf("A3:%x\n", A3 & leftover_mask);
            printf("A4:%x\n", A4 & leftover_mask);
            printf("B:%x\n",B1 & leftover_mask);

        }
        accum1 += POPCOUNT(XNOR(B1,A1) & leftover_mask);
        accum2 += POPCOUNT(XNOR(B1,A2) & leftover_mask);
        accum3 += POPCOUNT(XNOR(B1,A3) & leftover_mask);
        accum4 += POPCOUNT(XNOR(B1,A4) & leftover_mask);


        if(DEBUG) {
            printf("ACCUM:%d\n", accum1);
            printf("ACCUM2:%d\n", accum2);
            printf("ACCUM3:%d\n", accum3);
            printf("ACCUM4:%d\n", accum4);
        }

        //THRESHOLDING AND BINARIZATION
        if(accum1 >= *(thresholds++)) {
            z |= out_mask;
        }
        out_mask = ROTR(out_mask,1);
        if(accum2 >= *(thresholds++)) {
            z |= (out_mask);
        }
        out_mask = ROTR(out_mask,1);
        if(accum3 >= *(thresholds++)) {
            z |= (out_mask);
        }
        out_mask = ROTR(out_mask,1);
        if(accum4 >= *(thresholds++)) {
            z |= (out_mask);
        }
        out_mask = ROTR(out_mask,1);


        weights_overstep1  = MODULO_32(weights_overstep4  + dim_ker);

        A1 = A4_next;
        A1 <<= weights_overstep1;
        int next_weights_in_next_word = ((weights_overstep1 + words_leftover_kernel) >=32) | next_pA;
        pA1_next =pA4_next + next_weights_in_next_word;
        A1_next = *(pA1_next);
        weights_mask1 = (1<< (weights_overstep1)) - 1;
        A1 = A1 |  (A1_next >> (32-weights_overstep1) & weights_mask1);

        pA1 = pA4_next;


        //TODO Find a smarter way to do this
        //TODO We handle only output channels power of 2
        savecounter +=4;
        if((savecounter)%32==0) {
            *(pOut++)|=(z>>output_offset);
            savecounter=0;
            z = 0;
        }

    }



    while(leftover_channels) {
        B1 = *(pB1);
        pB1_next = pB1 + 1;
        B1_next = *(pB1_next);
        B1 <<= input_offset;
        uint32_t input_mask = (1<< input_offset) - 1;
        B1= B1| (B1_next >> (32  - input_offset) & input_mask);


        //Store the conv output
        accum1 =0;



        for (int words_in = 0; words_in < words_kernel; words_in++) {

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
            A1 <<= weights_overstep1;
            //A2 deve sempre puntare alla parola con i prossimi pesi necessari al seguente calcolo
            uint32_t o = ((words_leftover_kernel + weights_overstep1) >= 32) | ( words_in != (words_kernel - 1));
            pA1_next += o;
            A1_next = (*(pA1_next));
            A1 = A1 | ( (A1_next >> (32-weights_overstep1)) & weights_mask1);

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


        weights_overstep1 = MODULO_32(weights_overstep1 + dim_ker);

        //Next set of weights
        A1 = A1_next;
        A1 <<= weights_overstep1;
        pA1_next += ((weights_overstep1 + dim_ker) >=32) | (next_pA);
        A1_next = *(pA1_next);
        weights_mask1 = (1<< (weights_overstep1)) - 1;
        A1 = A1 |  (A1_next >> (32-weights_overstep1) & weights_mask1);


        //TODO Find a smarter way to do this
        savecounter +=1;
        out_mask = ROTR(out_mask,1);
        if((savecounter)%32==0) {
            *(pOut++)|=(z>>output_offset);
            savecounter=0;
            z = 0;
        }

    leftover_channels--;
    }
    if(z!=0)
    {
        *(pOut) |= (z>>output_offset);
        pOut += ((output_offset+ch_out)==32);

    }

    return pOut;


}
