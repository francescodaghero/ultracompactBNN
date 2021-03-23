PULP_APP = main

#Test Nx1
#PULP_APP_FC_SRCS =src/main.c
#PULP_APP_FC_SRCS +=src/convolution1bit_Nx1.c
#PULP_APP_FC_SRCS +=src/xnorpop4x1.c
#PULP_APP_FC_SRCS +=src/xnorpop2x1.c
#PULP_APP_FC_SRCS +=src/xnorpop1x1.c

#Test NxM
PULP_APP_FC_SRCS =src/main.c
PULP_APP_FC_SRCS +=src/convolution1bit_NxM_pooling.c
PULP_APP_FC_SRCS +=src/xnorpop4x1_pooling.c
PULP_APP_FC_SRCS +=src/xnorpop2x2_pooling.c
#PULP_APP_FC_SRCS +=src/xnorpop2x1g.c
#PULP_APP_FC_SRCS +=src/xnorpop1x1g.c
#PULP_APP_FC_SRCS +=src/xnorpop4x2g.c
#PULP_APP_FC_SRCS +=src/xnorpop1x2g.c
#PULP_APP_FC_SRCS +=src/xnorpop4x4g.c
#PULP_APP_FC_SRCS +=src/xnorpop2x4g.c
#PULP_APP_FC_SRCS +=src/xnorpop1x4g.c



cores=1
disopt =--disassembler-options="march=rv32imfcxpulpv2"
#disopt =-dsrwCS

PULP_CFLAGS += -DNUM_CORES=$(CORES)  -O3 -w

ifdef memory
PULP_CFLAGS += -DCODESIZE=1
endif



#PULP_CFLAGS += -DPROFILING
PULP_LDFLAGS = -flto
PULP_CFLAGS += -DSTATS #Aggiunto per il profiling
PULP_CFLAGS += -flto #Aggiunto da me
#PULP_CFLAGS += -fno-tree-vectorize -fno-tree-loop-distribution -fno-tree-loop-distribute-patterns
#PULP_CFLAGS += -fno-unroll-loops -fno-unswitch-loops

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
