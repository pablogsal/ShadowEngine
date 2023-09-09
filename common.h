#pragma once

// Pi and Pi/2 constants
#include <math.h>
#define Pi M_PI
#define HALF_PI 1.57079632679489655799898173427209258079528808593750

// Definition of the data type
typedef float Real;

// Define a structure to store the black hole and Kerr constants
struct BlackHoleConstants {
    Real a;
    Real a2;

    Real ro;         
    Real delta;      
    Real sigma;
    Real alpha;     
    Real omega;     
    Real pomega;     
};

// Define a structure to store the camera constants
struct Camera {
    Real r;       
    Real theta;   
    Real phi;     

    Real roll;
    Real pitch;
    Real yaw;
    
    Real focal_distance;

    Real beta;

    Real pixel_width;
    Real pixel_height;
};

// Declaration of the system size; i.e., the number of equations
constexpr int SYSTEM_SIZE = 5;
constexpr int DATA_SIZE = 2;

// Declaration of the image parameters: number of rows and columns, as well as
// the total amount of pixels.
constexpr int IMG_ROWS = 500;
constexpr int IMG_COLS = 500;
constexpr int NUM_PIXELS = IMG_ROWS * IMG_COLS;

// Convention for ray's status
typedef enum rayStatus {
    SPHERE = 0,
    DISK = 1,
    HORIZON = 2,
} RayStatus;

// Black hole parameters: horizon radius and disk definition
constexpr int horizonRadius = 1.96432165911;
constexpr int innerDiskRadius = 0;
constexpr int outerDiskRadius = 20;

// Enumerate to make the communication between SolverRK4(5) and its callers easier
typedef enum solverStatus{
    SOLVER_SUCCESS,
    SOLVER_FAILURE
} SolverStatus;

/**
 * Returns the sign of `x`; i.e., it returns +1 if x >= 0 and -1 otherwise.
 * @param  x The number whose sign has to be returned
 * @return   Sign of `x`, considering 0 as positive.
 */
__device__ inline int sign(Real x){
    return x < 0 ? -1 : +1;
}