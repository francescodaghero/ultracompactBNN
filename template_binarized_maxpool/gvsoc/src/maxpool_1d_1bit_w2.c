//#include "stdint.h"
//#include "stdio.h"
#include "rt/rt_api.h"
#include "../include/kernels.h"

//MACROS for pulp
//#define POPCOUNT(x) __builtin_popcount(x)
//#define ROTR(x,bits) rotate_right(x,bits)
#define POPCOUNT(x) __builtin_pulp_cnt(x)
#define ROTR(x,bits) __builtin_pulp_rotr(x,bits)


//MACROS for fast division and modulus
#define DIVIDE_32(x) x>>5
#define MODULO_32(x) (x & 0x1F)
#define XNOR(B,A) (~(B ^ A))

//#define DEBUG 0


//static inline uint32_t rotate_right (uint32_t u, size_t r)
//{
//    __asm__ ("rorl %%cl, %0" : "+r" (u) : "c" (r));
//    return u;
//}


void __attribute__((always_inline)) maxpool_1d_1bit_w2_fullstrided(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint16_t  ch_in,
        const uint16_t  kernel_size,
        const uint16_t  stride,
        uint32_t *       pOutBuffer,
        const uint16_t  dim_out
)
{

    if((ch_in & (ch_in - 1)) != 0) {
        return;
    }


    uint32_t offset = DIVIDE_32((ch_in +31));

    uint32_t * pA = pInBuffer;
    uint32_t * pB;
    uint32_t *pOut    = pOutBuffer;
    uint32_t B;

    uint32_t mask = ch_in < 32 ? (1<<ch_in) -1 : 0xffffffff;
    uint32_t dim = DIVIDE_32((dim_out*ch_in+31));
    uint32_t rot = 32 - ch_in;
    uint32_t n_words = DIVIDE_32((kernel_size*ch_in)); //Caso 64 ch-in, qua ho 6 parole totali
    uint32_t leftover = MODULO_32(kernel_size*ch_in);

    //Additional stride handling variables
    uint32_t jumper = 0; //Variable 0 if we have just to go to the next word, 1 if we have to jump
    uint32_t stride_offset=0; //Counter to understand if have to jump at a given word
    uint32_t stride_jump = (stride * (ch_in/32) - ch_in/32); //The actual jump forward
    uint32_t kernel_offset = ch_in <=32 ? 1 : (ch_in/32); //The number of words after we have to skip E.G 2 for 64, 3 for 128


    uint32_t full_dim_out = (1 + dim_out)*stride - 1;
    uint32_t dim_stride1 = DIVIDE_32((full_dim_out*ch_in+31));
    int ts=0, gl_ts_shift =0;
    uint32_t out = 0;
    uint32_t mask_stride = ~((1<< (32-ch_in)) -1);
    int dim_out_tmp = dim_out;
    int enable_stride = stride!=1 && ch_in<32;
    int input_jump = (ch_in*stride+31)>>5;

    for (int i = 0; i < dim_stride1; i+=input_jump) {
        uint32_t A = *(pA);
        pB = pA + offset; //Lavoro con la parola dopo se <64, se no diventa ch_in/32
        B = *(pB);

        //Additional computations for stride
        stride_offset +=1;
        jumper=stride_offset >= kernel_offset; //stride_offset/(kernel_offset);

        stride_offset %= (kernel_offset);

        pA+= (ch_in*stride+31)>>5;//;jumper * (stride_jump)  + 1;
        uint32_t z = 0;

        //TODO Divide n_words to use unitary increment, perform div using builtin logarithm
        for (int j = 0; j < n_words; j += offset) {
            B = *(pB);

            pB = pB + offset;

            for (int k = 0; k < 32; k += ch_in) {
                z = z | A;
                A <<= ch_in; //ch_in > 32 -> undefined but not used
                //TODO Avoid this operation, replace above shift with pulp builtin
                A &= (~mask); //Useless for ch-in <32, clears A for ch_in > 32, see shift <<32
                B = ROTR(B, rot);
                A = A | (B & mask);
            }
        }
        for (int k = 0; k < leftover; k += ch_in) {
            z = z | A;
            A <<= ch_in;
            B = ROTR(B, rot);
            A = A | (B & mask);
        }


        if(enable_stride) {
            while (ts < 32 && dim_out_tmp) {
                out |= ((z << ts) >> gl_ts_shift) & mask_stride;
                mask_stride = ROTR(mask_stride, ch_in);
                ts += ch_in * stride;
                gl_ts_shift += ch_in;
//                if (gl_ts_shift >> 5) {
//                    *(pOut++) = out;
//                    out = 0;
//                }
//                gl_ts_shift %= 32;
                dim_out_tmp--;

            }
            if (gl_ts_shift >> 5) {
                    *(pOut++) = out;
                    out = 0;
                }
            gl_ts_shift %= 32;
            ts %= 32;
        } else {
            *(pOut++) = z;
        }
    }

    if(enable_stride && out) {
        *pOut=out;
    }

}

#ifdef TEST_FUNCTIONS
void maxpool_1d_1bit_w2(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint16_t  ch_in,
        const uint16_t  kernel_size,
        const uint16_t  stride,
        uint32_t *       pOutBuffer,
        const uint16_t  dim_out
)
{

    if((ch_in & (ch_in - 1)) != 0 || stride != 1) {
        return;
    }

    uint32_t offset = DIVIDE_32((ch_in +31));

    uint32_t * pA = pInBuffer;
    uint32_t * pB;
    uint32_t *pOut    = pOutBuffer;
    uint32_t B;

    uint32_t mask = ch_in < 32 ? (1<<ch_in) -1 : 0xffffffff;
    uint32_t dim = DIVIDE_32((dim_out*ch_in+31));
    uint32_t rot = 32 - ch_in;
    uint32_t n_words = DIVIDE_32((kernel_size*ch_in)); //Caso 64 ch-in, qua ho 6 parole totali
    uint32_t leftover = MODULO_32(kernel_size*ch_in);


        for (int i = 0; i < dim; i++) {
            uint32_t A = *(pA);
            pB = pA + offset; //Lavoro con la parola dopo se <64, se no diventa ch_in/32
            B = *(pB);
            pA++;
            uint32_t z = 0;

            //TODO Divide n_words to use unitary increment, perform div using builtin logarithm
            for (int j = 0; j < n_words; j+=offset) {
                B = *(pB);

                pB = pB +offset;

                for (int k = 0; k < 32; k += ch_in) {
                    z = z | A;
                    A <<= ch_in; //ch_in > 32 -> undefined but not used
                    //TODO Avoid this operation, replace above shift with pulp builtin
                    A &= (~mask); //Useless for ch-in <32, clears A for ch_in > 32, see shift <<32
                    B = ROTR(B, rot);
                    A = A | (B & mask);
                }
            }
            for (int k = 0; k < leftover; k += ch_in) {
                z = z | A;
                A <<= ch_in;
                B = ROTR(B, rot);
                A = A | (B & mask);
            }
            *(pOut++) = z;
        }


}

//TODO Improve this function
//Handles any stride for channels >=32
void maxpool_1d_1bit_w2_strided(
        const uint32_t * pInBuffer,
        const uint16_t  dim_in,
        const uint16_t  ch_in,
        const uint16_t  kernel_size,
        const uint16_t  stride,
        uint32_t *       pOutBuffer,
        const uint16_t  dim_out
)
{

    if((ch_in & (ch_in - 1)) != 0) {
        return;
    }

    uint32_t offset = DIVIDE_32((ch_in +31));

    uint32_t * pA = pInBuffer;
    uint32_t * pB;
    uint32_t *pOut    = pOutBuffer;
    uint32_t B;

    uint32_t mask = ch_in < 32 ? (1<<ch_in) -1 : 0xffffffff;
    uint32_t dim = DIVIDE_32((dim_out*ch_in+31));
    uint32_t rot = 32 - ch_in;
    uint32_t n_words = DIVIDE_32((kernel_size*ch_in)); //Caso 64 ch-in, qua ho 6 parole totali
    uint32_t leftover = MODULO_32(kernel_size*ch_in);

    //Additional stride handling variables
    uint32_t jumper = 0; //Variable 0 if we have just to go to the next word, 1 if we have to jump
    uint32_t stride_offset=0; //Counter to understand if have to jump at a given word
    uint32_t stride_jump = (stride * (ch_in/32) - ch_in/32); //The actual jump forward
    uint32_t kernel_offset = ch_in <=32 ? 1 : (ch_in/32); //The number of words after we have to skip E.G 2 for 64, 3 for 128

    for (int i = 0; i < dim; i++) {
        uint32_t A = *(pA);
        pB = pA + offset; //Lavoro con la parola dopo se <64, se no diventa ch_in/32

        //Additional computations for stride
        stride_offset +=1;
        jumper=stride_offset >= kernel_offset; //stride_offset/(kernel_offset);

        stride_offset %= (kernel_offset);

        pA+= jumper * (stride_jump)  + 1;
        uint32_t z = 0;

        //TODO Divide n_words to use unitary increment, perform div using builtin logarithm
        for (int j = 0; j < n_words; j += offset) {
            B = *(pB);

            pB = pB + offset;

            for (int k = 0; k < 32; k += ch_in) {
                z = z | A;
                A <<= ch_in; //ch_in > 32 -> undefined but not used
                //TODO Avoid this operation, replace above shift with pulp builtin
                A &= (~mask); //Useless for ch-in <32, clears A for ch_in > 32, see shift <<32
                B = ROTR(B, rot);
                A = A | (B & mask);
            }
        }
        for (int k = 0; k < leftover; k += ch_in) {
            z = z | A;
            A <<= ch_in;
            B = ROTR(B, rot);
            A = A | (B & mask);
        }
        *(pOut++) = z;
    }

}
#endif




