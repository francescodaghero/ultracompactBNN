#ifndef _STATS_H
#define _STATS_H

#define HOTTING 2
#define REPEAT  3

#ifdef STATS

#define INIT_STATS() \
    rt_perf_t perf; \
    unsigned long _cycles = 0; \
    unsigned long _instr = 0; \
    unsigned long _active = 0; \
    unsigned long _ldext = 0; \
    unsigned long _tcdmcont = 0; \
    unsigned long _ldstall = 0; \
    unsigned long _imiss = 0;

#define BEGIN_STATS_LOOP()  \
    for(int _k=0; _k<HOTTING+REPEAT; _k++) { \
      rt_perf_init(&perf); \
      rt_perf_conf(&perf,(1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR) | (1<<RT_PERF_ACTIVE_CYCLES) | (1<<RT_PERF_LD_EXT) | (1<<RT_PERF_TCDM_CONT) | (1<<RT_PERF_LD_STALL) | (1<<RT_PERF_IMISS) );


#define START_STATS()  \
    rt_perf_reset(&perf); \
    rt_perf_start(&perf);

#define STOP_STATS() \
   rt_perf_stop(&perf); \
   rt_perf_save(&perf); \
   if (_k >= HOTTING) \
    { \
      _cycles   += rt_perf_read (RT_PERF_CYCLES); \
      _instr    += rt_perf_read (RT_PERF_INSTR); \
    	_active   += rt_perf_read (RT_PERF_ACTIVE_CYCLES); \
      _ldext    += rt_perf_read (RT_PERF_LD_EXT); \
    	_tcdmcont += rt_perf_read (RT_PERF_TCDM_CONT); \
    	_ldstall  += rt_perf_read (RT_PERF_LD_STALL); \
      _imiss    += rt_perf_read (RT_PERF_IMISS); \
    }

#define END_STATS_LOOP()  \
   if (_k == HOTTING+REPEAT-1) \
    { \
      int id = rt_core_id(); \
      printf("[%d] cycles = %lu\n", id, _cycles/REPEAT); \
      printf("[%d] instr = %lu\n", id, _instr/REPEAT); \
    	printf("[%d] active cycles = %lu\n", id, _active/REPEAT); \
      printf("[%d] ext load = %lu\n", id, _ldext/REPEAT); \
    	printf("[%d] TCDM cont = %lu\n", id, _tcdmcont/REPEAT); \
    	printf("[%d] ld stall = %lu\n", id, _ldstall/REPEAT); \
    	printf("[%d] imiss = %lu\n", id, _imiss/REPEAT); \
    } \
  }

#else // STATS

#define INIT_STATS()
#define BEGIN_STATS_LOOP()
#define START_STATS()
#define STOP_STATS()
#define END_STATS_LOOP()

#endif  // STATS

#endif
