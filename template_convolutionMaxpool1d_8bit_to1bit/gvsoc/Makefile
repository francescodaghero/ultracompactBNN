PULP_APP = main


PULP_APP_FC_SRCS =src/main.c
PULP_APP_FC_SRCS +=src/conv1bit_8bit_pooling.c
PULP_APP_FC_SRCS +=src/matmul4x2_pooling.c
PULP_APP_FC_SRCS +=src/matmul4x1_pooling.c



cores=1

PULP_CFLAGS += -DNUM_CORES=$(cores)  -flto -O3
ifdef memory
PULP_CFLAGS += -DCODESIZE=1
endif
PULP_LDFLAGS += -flto
PULP_CFLAGS += -DSTATS


include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
