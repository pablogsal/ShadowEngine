#pragma once

#include "common.h"
/**
* Computes the value of the function F(t) = (dr/dt, dtheta/dt, dphi/dt, dpr/dt,
* dptheta/dt) and stores it in the memory pointed by f.
 * @param[in]  y         Initial conditions for the system: a pointer to
 *                       a vector whose lenght shall be the same as the
 *                       number of equations in the system: 5
 * @param[in]  f         Computed value of the function: a pointer to a
 *                       vector whose lenght shall be the same as the
 *                       number of equations in the system: 5
 * @param[in]  data      Additional data needed by the function, managed
 *                       by the caller. Currently used to get the ray's
 *                       constants b and q.
 */
__device__ void computeComponent(BlackHoleConstants bh, Real* y, Real* f, Real* data);