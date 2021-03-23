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
#define XNOR(B,A) (~(B ^ A))


void xnorpop3x1(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint32_t *  pWeight,
        const uint16_t  dim_out,
        int32_t *       pOutBuffer,
        const int32_t *thresholds)
{

    int32_t *pOut    = pOutBuffer;

    uint32_t B;
    uint32_t *pB = pInBuffer;
    uint32_t *pB2;

    uint32_t A1, A1_next;
    uint32_t *pA1_next, *pA1;

    uint32_t A2, A2_next;
    uint32_t *pA2_next, *pA2;

    uint32_t A3, A3_next;
    uint32_t *pA3_next, *pA3;


    int feature_out;
    int output_features_leftover= dim_out%3;
    uint32_t dim_feat = dim_in;
    uint32_t next_pA = dim_feat>=32;
    uint32_t words_infeatures_leftover = MODULO_32(dim_feat);
    uint32_t words_infeatures = DIVIDE_32(dim_feat);
    //0xxxx00000 mask with leftover bits set AT THE BEGINNING
    uint32_t leftover_mask = words_infeatures_leftover!=0 ? ~((1 << (32 - words_infeatures_leftover )) - 1) : 0x0;
    uint32_t next_weight_set = words_infeatures >=1 ? (words_infeatures - 1) : 0;
    //Original set of weights
    pA1 = pWeight;
    A1 = (*pA1);
    pA1_next = pA1 + (next_pA);
    A1_next = *(pA1_next);
    uint32_t weights_overstep1 =0;
    int16_t accum1;
    int16_t accum2;
    int16_t accum3;




    uint32_t weights_mask1 = (1<< (weights_overstep1)) - 1;
    uint32_t weights_mask2, weights_mask3;
    int thr_feature_index = 0;

    //Iterating over output features/classes
    for(feature_out = 0; feature_out< dim_out/3; feature_out+=1) {
        //Iterazioni sui canali in output

        //Successive sets of weights
        uint32_t weights_overstep2 = MODULO_32(weights_overstep1+words_infeatures_leftover);
        pA2 = pA1 + words_infeatures + ((weights_overstep1+words_infeatures_leftover) >=32); //Point pA2 all'inizio dei pesi
        int next_weights_in_next_word2 = ((weights_overstep2 + words_infeatures_leftover) >=32) | next_pA;
        pA2_next = pA2 + next_weights_in_next_word2;
        A2 = (*(pA2)) << weights_overstep2;
        A2_next = *(pA2_next);
        weights_mask2 = (1<< (weights_overstep2)) - 1;
        A2 = A2 |  (A2_next >> (32-weights_overstep2) & weights_mask2);


        //Successive sets of weights
        uint32_t weights_overstep3 = MODULO_32(weights_overstep2+words_infeatures_leftover);
        pA3 = pA2 + words_infeatures + ((weights_overstep2+words_infeatures_leftover) >=32); //Point pA2 all'inizio dei pesi
        int next_weights_in_next_word3 = ((weights_overstep3 + words_infeatures_leftover) >=32) | next_pA;
        pA3_next = pA3 + next_weights_in_next_word3;
        A3 = (*(pA3)) << weights_overstep3;
        A3_next = *(pA3_next);
        weights_mask3 = (1<< (weights_overstep3)) - 1;
        A3 = A3 |  (A3_next >> (32-weights_overstep3) & weights_mask3);


#ifdef DEBUG
        printf("-------------------------\n");
#endif
        //Input restarts always from the same timestep
        B = *(pB);
        pB2 = pB;

        //Store the conv output
        accum1 =0;
        accum2 = 0;
        accum3  = 0;
        for (int words_in = 0; words_in < words_infeatures; words_in++) {

            accum1 += POPCOUNT(XNOR(B,A1));
            accum2 += POPCOUNT(XNOR(B,A2));
            accum3 += POPCOUNT(XNOR(B,A3));



#ifdef DEBUG
            printf("acc: %d\n",accum);
                            printf("A1:%x\n", A1);
                            printf("B:%x\n",B);
#endif

            //Next words are already loaded
            B = *(++pB2);

            //int is_not_last_iteration = ( words_in != (words_infeatures - 1))


            A1 = A1_next;
            A1 <<= weights_overstep1;
            //A1next deve sempre puntare alla parola con i prossimi pesi necessari al seguente calcolo
            uint32_t o = ((words_infeatures_leftover + weights_overstep1) >= 32) | ( words_in != (words_infeatures - 1));
            pA1_next += o;
            A1_next = (*(pA1_next));
            A1 = A1 | ( (A1_next >> (32-weights_overstep1)) & weights_mask1);

            A2 = A2_next;
            A2 <<= weights_overstep2;
            uint32_t incr = ((words_infeatures_leftover + weights_overstep2) >= 32) | ( words_in != (words_infeatures - 1));
            pA2_next += incr;
            A2_next = (*(pA2_next));
            A2 = A2 | ( (A2_next >> (32-weights_overstep2)) & weights_mask2);

            A3 = A3_next;
            A3 <<= weights_overstep3;
            uint32_t incr3 = ((words_infeatures_leftover + weights_overstep3) >= 32) | ( words_in != (words_infeatures - 1));
            pA3_next += incr3;
            A3_next = (*(pA3_next));
            A3 = A3 | ( (A3_next >> (32-weights_overstep3)) & weights_mask3);

        }


        //ODD KERNELS or DIM_KER < 32

#ifdef DEBUG
        uint32_t mul = ~((B ^ A1)) & leftover_mask;
                uint32_t mul2 = ~((B ^ A2)) & leftover_mask;
                printf("A1:%x\n", A1);
                printf("B:%x\n",B);
#endif

        accum1 += POPCOUNT(XNOR(B,A1) & leftover_mask);
        accum2 += POPCOUNT(XNOR(B,A2) & leftover_mask);
        accum3 += POPCOUNT(XNOR(B,A3) & leftover_mask);

#ifdef DEBUG
        printf("%d\n", 2*accum1-dim_in);
                printf("%d\n", 2*accum2-dim_in);

#endif

        //COMPUTING FINAL OUTPUT

        *(pOut++) =(int32_t) (accum1 * thresholds[thr_feature_index++] + thresholds[thr_feature_index++]);
        *(pOut++) =(int32_t) (accum2 * thresholds[thr_feature_index++] + thresholds[thr_feature_index++]);
        *(pOut++) =(int32_t) (accum3 * thresholds[thr_feature_index++] + thresholds[thr_feature_index++]);


        weights_overstep1 = MODULO_32((weights_overstep3 + dim_feat));


        //Next set of weights, equate to the last feature calculated
        A1 = A3_next;
        A1 <<= weights_overstep1;
        int next_weights_in_next_word = ((weights_overstep1 + words_infeatures_leftover) >=32) | next_pA;
        pA1_next =pA3_next + next_weights_in_next_word;
        A1_next = *(pA1_next);
        weights_mask1 = (1<< (weights_overstep1)) - 1;
        A1 = A1 |  (A1_next >> (32-weights_overstep1) & weights_mask1);

        pA1 = pA3_next;

    }

    //Features leftover
    while(output_features_leftover) {

#ifdef DEBUG
        printf("-------------------------\n");
#endif
        //Input restarts always from the same timestep
        B = *(pB);
        pB2 = pB;

        //Store the conv output
        accum1 =0;
        for (int words_in = 0; words_in < words_infeatures; words_in++) {

            accum1 += POPCOUNT(XNOR(B,A1));

#ifdef DEBUG
            printf("acc: %d\n",accum);
                            printf("A1:%x\n", A1);
                            printf("B:%x\n",B);
#endif

            //Next words are already loaded
            B = *(++pB2);


            A1 = A1_next;
            A1 <<= weights_overstep1;
            //A2 deve sempre puntare alla parola con i prossimi pesi necessari al seguente calcolo
            uint32_t o = ((words_infeatures_leftover + weights_overstep1) >= 32) | ( words_in != (words_infeatures - 1));
            pA1_next += o;
            A1_next = (*(pA1_next));
            A1 = A1 | ( (A1_next >> (32-weights_overstep1)) & weights_mask1);

        }


        //ODD KERNELS or DIM_KER < 32

#ifdef DEBUG
        uint32_t mul = ~((B ^ A1)) & leftover_mask;
                        uint32_t mul2 = ~((B ^ A2)) & leftover_mask;
                        printf("A1:%x\n", A1);
                        printf("B:%x\n",B);
#endif

        accum1 += POPCOUNT(XNOR(B,A1) & leftover_mask);
#ifdef DEBUG
        printf("%d\n", 2*accum1-dim_in);
#endif

        //COMPUTING FINAL OUTPUT

        *(pOut++) =(int32_t) (accum1 * thresholds[thr_feature_index++] + thresholds[thr_feature_index++]);

        weights_overstep1 = MODULO_32((weights_overstep1 + dim_feat));


        //Next set of weights, equate to the last feature calculated
        A1 = A1_next;
        A1 <<= weights_overstep1;
        pA1_next += ((weights_overstep1 + words_infeatures_leftover) >=32) | next_pA;
        A1_next = *(pA1_next);
        weights_mask1 = (1<< (weights_overstep1)) - 1;
        A1 = A1 |  (A1_next >> (32-weights_overstep1) & weights_mask1);


        output_features_leftover--;
    }


}
