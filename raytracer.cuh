#include "common.h"


/**
 * CUDA kernel that computes the initial conditions (r, theta, phi, pR, pPhi)
 * and the constants (b, q) of every ray in the simulation.
 *
 * This method depends on the shape of the CUDA grid: it is expected to be a 2D
 * matrix with at least IMG_ROWS threads in the Y direction and IMG_COLS
 * threads in the X direction. Every pixel of the camera is assigned to a
 * single thread that computes the initial conditions and constants of its
 * corresponding ray, following a pinhole camera model.
 *
 * Each thread that executes this method implements the following algorithm:
 * 		1. Compute the pixel physical coordinates, considering the center of
 * 		the sensor as the origin and computing the physical position using the
 * 		width and height of each pixel.
 * 		2. Compute the ray's incoming direction, theta and phi, on the camera's
 * 		local sky, following the pinhole camera model defined by the sensor
 * 		shape and the focal distance __d.
 * 		3. Compute the canonical momenta pR, pTheta and pPhi with the method
 * 		`getCanonicalMomenta`.
 * 		4. Compute the ray's constants b and q with the method.
 * 		`getConservedQuantities`.
 * 		5. Fill the pixel's corresponding entry in the global array pointed by
 * 		devInitCond with the initial conditions: __camR, __camTheta, __camPhi,
 * 		pR and pTheta, where the three first components are constants that
 * 		define the position of the focal point on the black hole coordinate
 * 		system.
 * 		6. Fill the pixel's corresponding entry in the global array pointed by
 * 		devConstants with the computed constants: b and q.
 *
 * @param[out]     devInitCond  Device pointer to a serialized 2D matrix
 *                 where each entry corresponds to a single pixel in the
 *                 camera sensor. If the sensor has R rows and C columns,
 *                 the vector pointed by devInitCond contains R*C
 *                 entries, where each entry is a 5-tuple prepared to
 *                 receive the initial conditions of a ray: (r, theta,
 *                 phi, pR, pPhi). At the end of this kernel, the array
 *                 pointed by devInitCond is filled with the initial
 *                 conditions of every ray.
 * @param[out]     devConstants  Device pointer to a serialized 2D matrix
 *                 where each entry corresponds to a single pixel in the
 *                 camera sensor. If the sensor has R rows and C columns,
 *                 the vector pointed by devConstants contains R*C
 *                 entries, where each entry is a 2-tuple prepared to
 *                 receive the constants of a ray: (b, q). At the end of
 *                 this kernel, the array pointed by devConstants is
 *                 filled with the computed constants of every ray.
 * @param[in]      pixelWidth   Width, in physical units, of the camera's
 *                 pixels.
 * @param[in]      pixelHeight  Height, in physical units, of the
 *                 camera's pixels.
 */
__global__ void setInitialConditions(CameraConstants camera, BlackHoleConstants bh,
                                     void* devInitCond,void* devConstants,
                                     Real pixelWidth, Real pixelHeight);

/**
 * CUDA kernel that integrates a set of photons backwards in time from x0 to
 * xend, storing the final results of their position and canonical momenta on
 * the array pointed by devInitCond.
 *
 * This method depends on the shape of the CUDA grid: it is expected to be a 2D
 * matrix with at least IMG_ROWS threads in the Y direction and IMG_COLS
 * threads in the X direction. Every ray is assigned to a single thread, which
 * computes its final state solving the ODE system defined by the relativistic
 * spacetime.
 *
 * Each thread that executes this method implements the following algorithm:
 * 		1. Copy the initial conditions and constants of the ray from its
 * 		corresponding position at the global array devInitCond and devData into
 * 		local memory.
 * 		2. Integrate the ray's equations defined in Thorne's paper, (A.15).
 * 		This is done while continuosly checking whether the ray has collided
 * 		with disk or horizon.
 * 		3. Overwrite the conditions at devInitCond to the new computed ones.
 * 		Fill the ray's final status (no collision, collision with the disk or
 * 		collision with the horizon) in the devStatus array.
 *
 * @param[in]       x0             Start of the integration interval
 *                  [x_0, x_{end}]. It is usually zero.
 * @param[in]       xend           End of the integration interval
 *                  [x_0, x_{end}].
 * @param[in,out]   devInitCond    Device pointer to a serialized 2D
 *                  Real matrix where each entry corresponds to a single
 *                  pixel in the camera sensor; i.e., to a single ray. If
 *                  the sensor has R rows and C columns, the vector
 *                  pointed by  devInitCond contains R*C entries, where
 *                  each entry is a 5-tuple filled with the initial
 *                  conditions of the corresponding ray: (r, theta, phi,
 *                  pR, pPhi). At the end of this kernel, the array
 *                  pointed by devInitCond is overwritten with the final
 *                  state of each ray.
 * @param[in]       h              Step size for the Runge-Kutta solver.
 * @param[in]       hmax           Value of the maximum step size allowed
 *                  in the Runge-Kutta solver.
 * @param[in]       devData        Device pointer to a serialized 2D
 *                  Real matrix where each entry corresponds to a single
 *                  pixel in the camera sensor; i.e., to a single ray. If
 *                  the sensor has R rows and C columns, the vector
 *                  pointed by devData contains R*C entries, where each
 *                  entry is a 2-tuple filled with the constants of the
 *                  corresponding ray: (b, q).
 * @param[out]      devStatus      Device pointer to a serialized 2D
 *                  Int matrix where each entry corresponds to a single
 *                  pixel in the camera sensor; i.e., to a single ray. If
 *                  the sensor has R rows and C columns, the vector
 *                  pointed by devData contains R*C entries, where each
 *                  entry is an integer that will store the ray's status
 *                  at the end of the kernel
 */
__global__ void kernel(BlackHoleConstants bh, Real x0, Real xend, void* devInitCond, Real h,
                       Real hmax, void* devData, void* devStatus);