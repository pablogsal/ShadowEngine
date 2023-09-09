#pragma once
#include <common.h>


 class RK45Solver{
    public:
         __device__ explicit RK45Solver(const BlackHoleConstants& bh): bh(bh) {};
         __device__ int solve(Real* globalX0, Real xend, Real* initCond, Real hOrig, Real hmax, Real* data, int* iterations);
    private:
        const BlackHoleConstants bh;

         const Real rtoli{1e-06};
         const Real atoli{1e-12};
         const Real safe{0.9};
          const Real safeInv{1.1111111111111112};
         const Real fac1{0.2};
         const Real fac1_inverse{5.0};
         const Real fac2{10.0};
         const Real fac2_inverse{0.1};
         const Real _beta{0.04};
         const Real uround{2.3e-16};
         const Real MAX_RESOL{-2.0};
         const Real MIN_RESOL{-0.1};

     __device__ Real advanceStep(Real* y0, Real h, Real* y1, Real* data);
     __device__ int bisect(Real* yOriginal, Real* data, Real step, Real x);
};
