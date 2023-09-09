#include <common.h>

 __device__ int SolverRK45(BlackHoleConstants bh, Real* globalX0, Real xend, Real* initCond,
                           Real hOrig, Real hmax, Real* data, int* iterations);