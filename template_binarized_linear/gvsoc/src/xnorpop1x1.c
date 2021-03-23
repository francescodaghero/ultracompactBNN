//TODO ADD a #define PULP to enable only macros for PULP and disable others
//Includes for pulp
#include "rt/rt_api.h"
#include <stdio.h>
#include <stdint.h>


#include "../include/kernels.h"



//MACROS for pulp
//#define POPCOUNT(x) __builtin_popcount(x)
//#define ROTR(x,bits) rotate_right(x,bits)
#define POPCOUNT(x) __builtin_pulp_cnt(x)
#define ROTR(x,bits) __pulp_builtin_rotr(x,y)

//MACROS for fast division and modulus
#define DIVIDE_32(x) x>>5
#define MODULO_32(x) (x & 0x1F)


//#define DEBUG

//TODO Turn into a macro
//Rotate
static inline uint32_t rotate_right (uint32_t u, size_t r)
{
    __asm__ ("rorl %%cl, %0" : "+r" (u) : "c" (r));
    return u;
}

//TODO Make this function handle both last linear layer (output probabilities) and general linear layer
void xnorpop1x1(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint32_t *  pWeight,
        const uint16_t  dim_out,
        int32_t *       pOutBuffer,
        const int32_t *thresholds
)
{

    int32_t *pOut    = pOutBuffer;

    uint32_t *pB = pInBuffer;
    uint32_t *pA;
    uint32_t *pB2;
    uint32_t *pA2;
    uint32_t B;
    uint32_t A, A2;

    int16_t accum;
    uint32_t dim_feat = dim_in;
    uint32_t next_pA = dim_feat>=32;
    uint32_t words_infeatures_leftover = MODULO_32(dim_feat);
    uint32_t words_infeatures = DIVIDE_32(dim_feat);
    //0xxxx00000 mask with leftover bits set AT THE BEGINNING
    uint32_t leftover_mask = words_infeatures_leftover!=0 ? ~((1 << (32 - words_infeatures_leftover )) - 1) : 0x0;


    pA = pWeight;
    A = (*pA);
    pA2 = pA + (next_pA);

    A2 = *(pA2);uint32_t weights_overstep =0;
    uint32_t weights_mask = (1<< (weights_overstep)) - 1;
    int thr_feature_index = 0;

    //Iterating over output channels/classes
    for(int feature_out = 0; feature_out< dim_out; feature_out++) {
        //Iterazioni sui canali in output

#ifdef DEBUG
        printf("-------------------------\n");
#endif
        //Input restarts always from the same timestep
        B = *(pB);
        pB2 = pB;

        //Store the conv output
        accum =0;
        for (int words_in = 0; words_in < words_infeatures; words_in++) {

            accum += POPCOUNT(~(B ^ A));

#ifdef DEBUG
            printf("acc: %d\n",accum);
                printf("A:%x\n", A);
                printf("B:%x\n",B);
#endif

            //Next words are already loaded
            B = *(++pB2);


            A = A2;
            A <<= weights_overstep;

            //A2 deve sempre puntare alla parola con i prossimi pesi necessari al seguente calcolo
            uint32_t o = ((words_infeatures_leftover + weights_overstep) >= 32) | ( words_in != (words_infeatures - 1));
            pA2 += o;

            A2 = (*(pA2));
            A = A | ( (A2 >> (32-weights_overstep)) & weights_mask);

        }


        //ODD KERNELS or DIM_KER < 32
        uint32_t mul = ~((B ^ A)) & leftover_mask;

#ifdef DEBUG
        printf("A:%x\n", A);
        printf("B:%x\n",B);
#endif

        accum += POPCOUNT(mul);

#ifdef DEBUG
        printf("%d\n", 2*accum-dim_in);
#endif

        //COMPUTING FINAL OUTPUT

        *(pOut++) =(int32_t) (accum * thresholds[thr_feature_index++] + thresholds[thr_feature_index++]);

        weights_overstep = MODULO_32((weights_overstep + dim_feat));

        //Next set of weights
        A = A2;
        A <<= weights_overstep;
        pA2 += ((weights_overstep + words_infeatures_leftover) >=32) | next_pA;
        A2 = *(pA2);
        weights_mask = (1<< (weights_overstep)) - 1;
        A = A |  (A2 >> (32-weights_overstep) & weights_mask);

    }


}



