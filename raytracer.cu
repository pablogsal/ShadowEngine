#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "common.h"
#include "solver.cuh"

#define Pi M_PI
#define SYSTEM_SIZE 5

/**
 * Given the ray's incoming direction on the camera's local sky, rayTheta and
 * rayPhi, this function computes its canonical momenta. See Thorne's paper,
 * equation (A.11), for more information.
 * Note that the computation of this quantities depends on the constants
 * __camBeta (speed of the camera) and __ro, __alpha, __omega, __pomega and
 * __ro  (Kerr metric constants), that are defined in the common.cu template.
 * @param[in]       rayTheta      Polar angle, or inclination, of the
 *                  ray's incoming direction on the camera's local sky.
 * @param[in]       rayPhi        Azimuthal angle, or azimuth, of the
 *                  ray's incoming direction on the camera's local sky.
 * @param[out]      pR            Computed covariant coordinate r of the
 *                  ray's 4-momentum.
 * @param[out]      pTheta        Computed covariant coordinate theta of
 *                  the ray's 4-momentum.
 * @param[out]      pPhi          Computed covariant coordinate phi of
 *                  the ray's 4-momentum.
 */
static __device__ void getCanonicalMomenta(CameraConstants camera, BlackHoleConstants bh, Real rayTheta, Real rayPhi, Real* pR, Real* pTheta, Real* pPhi){
    // **************************** SET NORMAL **************************** //
    // Cartesian components of the unit vector N pointing in the direction of
    // the incoming ray
    Real Nx = sin(rayTheta) * cos(rayPhi);
    Real Ny = sin(rayTheta) * sin(rayPhi);
    Real Nz = cos(rayTheta);

    // ********************** SET DIRECTION OF MOTION ********************** //
    // Compute denominator, common to all the cartesian components
    Real den = 1. - camera.beta * Ny;

    // Compute factor common to nx and nz
    Real fac = -sqrt(1. - camera.beta*camera.beta);

    // Compute cartesian coordinates of the direction of motion. See(A.9)
    Real nY = (-Ny + camera.beta) / den;
    Real nX = fac * Nx / den;
    Real nZ = fac * Nz / den;

    Real Br = 0;
    Real Btheta = 0;
    Real Bphi = 1;
    Real kappa = sqrt(1 - Btheta*Btheta);

    // Convert the direction of motion to the FIDO's spherical orthonormal
    // basis. See (A.10)
    Real nR =( Bphi * nX / kappa) + (Br * nY) + (Br*Btheta * nZ / kappa);
    Real nTheta = (Btheta * nY) - (kappa * nZ);
    Real nPhi = - (Br * nX / kappa) + (Bphi * nY) + (Bphi*Btheta * nZ / kappa);

    // *********************** SET CANONICAL MOMENTA *********************** //
    // Compute energy as measured by the FIDO. See (A.11)
    Real E = 1. / (bh.alpha + bh.omega * bh.pomega * nPhi);

    // Set conserved energy to unity. See (A.11)
    // Real pt = -1;

    // Compute the canonical momenta. See (A.11)
    *pR = E * bh.ro * nR / sqrt(bh.delta);
    *pTheta = E * bh.ro * nTheta;
    *pPhi = E * bh.pomega * nPhi;
}

/**
 * Given the ray's canonical momenta, this function computes its constants b
 * (the axial angular momentum) and q (Carter constant). See Thorne's paper,
 * equation (A.12), for more information.
 * Note that the computation of this quantities depends on the constant
 * __camTheta, which is the inclination of the camera with respect to the black
 * hole, and that is defined in the common.cu template
 * @param[in]       pTheta        Covariant coordinate theta of the ray's
 *                  4-momentum.
 * @param[in]       pPhi          Covariant coordinate phi of the ray's
 *                  4-momentum.
 * @param[out]      b             Computed axial angular momentum.
 * @param[out]      q             Computed Carter constant.
 */
static __device__ void getConservedQuantities(CameraConstants camera, BlackHoleConstants bh, Real pTheta, Real pPhi, Real* b, Real* q) {
    // ********************* GET CONSERVED QUANTITIES ********************* //
    // Compute axial angular momentum. See (A.12).
    *b = pPhi;

    // Compute Carter constant. See (A.12).
    Real sinT = sin(camera.theta);
    Real sinT2 = sinT*sinT;

    Real cosT = cos(camera.theta);
    Real cosT2 = cosT*cosT;

    Real pTheta2 = pTheta*pTheta;
    Real b2 = pPhi*pPhi;

    *q = pTheta2 + cosT2*((b2/sinT2) - bh.a2);
}


__global__ void setInitialConditions(CameraConstants camera, BlackHoleConstants bh,
                                     void* devInitCond,void* devConstants,
                                     Real pixelWidth, Real pixelHeight){
    // Each pixel is assigned to a single thread thorugh the grid and block
    // configuration, both of them being 2D matrices:
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // The blocks have always a multiple of 32 threads, configured in a 2D
    // shape. As it is possible that there are more threads than pixels, we
    // have to make sure that only the threads that have an assigned pixel are
    // running.
    if(row < IMG_ROWS && col < IMG_COLS){
        // Compute pixel unique identifier for this thread
        int pixel = row*IMG_COLS + col;

        // Compute the position in the global array to store the initial
        // conditions of this ray
        Real* globalInitCond = (Real*) devInitCond;
        Real* initCond = globalInitCond + pixel*SYSTEM_SIZE;

        // Compute the position in the global array to store the constants of
        // this ray
        Real* globalConstants = (Real*) devConstants;
        Real* constants = globalConstants + pixel*2;

        // Compute pixel position in physical units
        Real _x = - (col + 0.5 - IMG_COLS/2) * pixelWidth;
        Real _y = (row + 0.5 - IMG_ROWS/2) * pixelHeight;

        // Rotate the pixels with the roll angle
        Real x = _x * cos(camera.roll) - _y * sin(camera.roll);
        Real y = _x * sin(camera.roll) + _y * cos(camera.roll);

        // Compute direction of the incoming ray in the camera's reference
        // frame: we sum the yaw angle to phi and the pitch angle to theta in
        // order to implement the camera CCD rotation. See pitch, roll and yaw
        // attributes in Camera class (camera.py)
        Real rayPhi = camera.yaw + Pi + atan(x / camera.focal_distance);
        Real rayTheta = camera.pitch + Pi/2 + atan(y / sqrt(camera.focal_distance*camera.focal_distance + x*x));

        // Compute canonical momenta of the ray and the conserved quantites b
        // and q
        Real pR, pTheta, pPhi, b, q;
        getCanonicalMomenta(camera, bh, rayTheta, rayPhi, &pR, &pTheta, &pPhi);
        getConservedQuantities(camera, bh, pTheta, pPhi, &b, &q);

        // Save ray's initial conditions in the global array
        initCond[0] = camera.r;
        initCond[1] = camera.theta;
        initCond[2] = camera.phi;
        initCond[3] = pR;
        initCond[4] = pTheta;

        // Save ray's constants in the global array
        constants[0] = b;
        constants[1] = q;
    }
}

__global__ void kernel(BlackHoleConstants bh, Real x0, Real xend, void* devInitCond, Real h,
                       Real hmax, void* devData, void* devStatus){
    // Compute pixel's row and col of this thread
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // Only the threads that have a proper pixel shall compute its ray equation
    if(row < IMG_ROWS && col < IMG_COLS){
        // Compute pixel unique identifier for this thread
        int pixel = row*IMG_COLS + col;

        // Array of status flags: at the output, the (x,y)-th element will be
        // set to SPHERE, HORIZON or disk, showing the final state of the ray.
        int* globalStatus = (int*) devStatus;
        globalStatus += pixel;
        int status = *globalStatus;

        // Integrate the ray only if it's still in the sphere. If it has
        // collided either with the disk or within the horizon, it is not
        // necessary to integrate it anymore.
        if(status == SPHERE){
            // Retrieve the position where the initial conditions this block
            // will work with are.
            // Each block, absolutely identified in the grid by blockId, works
            // with only one initial condition (that has N elements, as N
            // equations are in the system). Then, the position of where these
            // initial conditions are stored in the serialized vector can be
            // computed as blockId * N.
            Real* globalInitCond = (Real*) devInitCond;
            globalInitCond += pixel * SYSTEM_SIZE;

            // Pointer to the additional data array used by computeComponent
            Real* globalData = (Real*) devData;
            globalData += pixel * DATA_SIZE;

            // Local arrays to store the initial conditions and the additional
            // data
            Real initCond[SYSTEM_SIZE], data[DATA_SIZE];

            // Retrieve the data from global to local memory :)
            memcpy(initCond, globalInitCond, sizeof(Real)*SYSTEM_SIZE);
            memcpy(data, globalData, sizeof(Real)*DATA_SIZE);

            // Current time
            Real x = x0;

            // Local variable to know how many iterations spent the solver in
            // the current step.
            int iterations = 0;

            // MAIN ROUTINE. Integrate the ray from x to xend, checking disk
            // collisions on the go with the following algorithm:
            //   -> 0. Check that the ray has not collided with the disk or
            //   with the horizon and that the current time has not exceeded
            //   the final time.
            //   -> 1. Advance the ray a step, calling the main RK45 solver.
            //   -> 2. Test whether the ray has collided with the horizon.
            //          2.1 If the answer to the 2. test is negative: test
            //          whether the current theta has crossed theta = pi/2,
            //          and call bisect in case it did, updating its status
            //          accordingly (set it to DISK if the ray collided with
            //          the horizon).
            //          2.2. If the answer to the 2. test is positive: update
            //          the status of the ray to HORIZON.
            status = SolverRK45(bh, &x, xend, initCond, h, xend - x, data, &iterations);

            // Update the global status variable with the new computed status
            *globalStatus = status;

            // And, finally, update the current ray state in global memory :)
            memcpy(globalInitCond, initCond, sizeof(Real)*SYSTEM_SIZE);
        } // If status == SPHERE

    } // If row < IMG_ROWS and col < IMG_COLS
}