PULP_APP = main


PULP_APP_FC_SRCS =src/main.c
PULP_APP_FC_SRCS +=src/linear.c
PULP_APP_FC_SRCS +=src/xnorpop4x1.c
#PULP_APP_FC_SRCS +=src/xnorpop3x1.c
#PULP_APP_FC_SRCS +=src/xnorpop2x1.c
#PULP_APP_FC_SRCS +=src/xnorpop1x1.c



cores=1
#disopt =--disassembler-options="march=rv32imfcxpulpv2"
disopt =-drwCS
PULP_CFLAGS += -DNUM_CORES=$(CORES)  -O3  -Iinc -w
ifdef memory
PULP_CFLAGS += -DCODESIZE=1
endif
#PULP_CFLAGS += -DPROFILING
PULP_LDFLAGS = -lm -flto
PULP_CFLAGS += -DSTATS #Aggiunto per il profiling
PULP_CFLAGS += -flto #Aggiunto da me
#PULP_CFLAGS += -DQ8

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
