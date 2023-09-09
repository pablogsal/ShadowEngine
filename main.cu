#include <iostream>
#include <fstream>

#include <cuda_runtime.h>

#include "raytracer.cuh"
#include "image_transformation.cuh"
#include "common.h"


// Constants
const Real CAM_SENSOR_HEIGHT = 16.0;
const Real CAM_SENSOR_WIDTH = 16.0;

const Real H = -0.001;
const Real HMAX = -150;

const Real X0 = 0;
const Real XEND = -150;

void writeRawData(const std::string& filename, const Real* data, int numRows, int numCols, int numChannels) {
    std::ofstream outputFile(filename, std::ios::binary);

    if (!outputFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int row = 0; row < numRows; ++row) {
        for (int col = 0; col < numCols; ++col) {
            for (int channel = 0; channel < numChannels; ++channel) {
                const Real value = data[(row * numCols + col) * numChannels + channel];
                outputFile.write(reinterpret_cast<const char*>(&value), sizeof(Real));
            }
        }
    }

    outputFile.close();
    std::cout << "Raw data written to " << filename << std::endl;
}

// Function to calculate KerrConstants
BlackHoleConstants calculateKerrConstants(Real a, const CameraConstants& camera) {
    BlackHoleConstants constants;
    constants.a = a;
    constants.a2 = a * a;

    Real r2 = camera.r * camera.r;
    Real a2 = a * a;
    // Calculate the constants
    constants.ro = sqrt(r2 + a2 * cos(camera.theta) * cos(camera.theta));
    constants.delta = r2 + a2;
    constants.sigma = sqrt((r2+a2)*(r2+a2) - a2 * constants.delta * sin(camera.theta) * sin(camera.theta));
    constants.alpha = constants.ro * sqrt(constants.delta) / constants.sigma;
    constants.omega = 2.0 * a * camera.r / (constants.sigma * constants.sigma);
    constants.pomega = constants.sigma * sin(camera.theta) / constants.ro;
    return constants;
}

struct InitialConditions {
    Real r;
    Real theta;
    Real phi;
    Real pR;
    Real pTheta;
};

int main() {
    // Compute the width and height of a pixel
    Real pixelWidth = CAM_SENSOR_WIDTH / static_cast<Real>(IMG_COLS);
    Real pixelHeight = CAM_SENSOR_HEIGHT / static_cast<Real>(IMG_ROWS);

    Real a = 0.2647333333333333;
    Real r = 40.0;
    Real theta = 1.413717694115407;
    Real phi = 0.0;

    Real roll = 0.0;
    Real pitch = 0.0;
    Real yaw = -0.06;

    // Initialize the black hole constants
    CameraConstants camera = {r, theta, phi, roll, pitch, yaw, r/3, 0.0};
    BlackHoleConstants kerrConstants = calculateKerrConstants(a, camera);

    std::cout << "Camera Constants:" << std::endl;
    std::cout << "r: " << camera.r << std::endl;
    std::cout << "theta: " << camera.theta << std::endl;
    std::cout << "phi: " << camera.phi << std::endl;
    std::cout << "roll: " << camera.roll << std::endl;
    std::cout << "pitch: " << camera.pitch << std::endl;
    std::cout << "yaw: " << camera.yaw << std::endl;

    std::cout << "Kerr Constants:" << std::endl;
    std::cout << "a: " << kerrConstants.a << std::endl;
    std::cout << "a2: " << kerrConstants.a2 << std::endl;
    std::cout << "ro: " << kerrConstants.ro << std::endl;
    std::cout << "delta: " << kerrConstants.delta << std::endl;
    std::cout << "sigma: " << kerrConstants.sigma << std::endl;
    std::cout << "alpha: " << kerrConstants.alpha << std::endl;
    std::cout << "omega: " << kerrConstants.omega << std::endl;
    std::cout << "pomega: " << kerrConstants.pomega << std::endl;

    // Print the computed pixel size
    std::cout << "Pixel Width: " << pixelWidth << std::endl;
    std::cout << "Pixel Height: " << pixelHeight << std::endl;

    // Initialize CUDA
    cudaSetDevice(0); // Select the CUDA device (change as needed).

    // Allocate memory for input and output data on the host.
    // You need to define data structures and allocate memory for devInitCond,
    // devConstants, devData, devStatus, devDiskTexture, devSphereTexture, devImage, etc.

    // Allocate the systemState array on the device
    Real* devSystemState = nullptr;
    cudaMalloc((void**)&devSystemState, IMG_ROWS * IMG_COLS * SYSTEM_SIZE * sizeof(Real));

    // Allocate the constants array on the device
    Real* devConstants = nullptr;
    cudaMalloc((void**)&devConstants, IMG_ROWS * IMG_COLS * 2 * sizeof(Real));

    // Allocate the rayStatus array on the device
    int* devRayStatus = nullptr;
    cudaMalloc((void**)&devRayStatus, IMG_ROWS * IMG_COLS * sizeof(int));

    // Check for CUDA allocation errors (optional)
    if (devSystemState == nullptr || devConstants == nullptr || devRayStatus == nullptr) {
        std::cerr << "CUDA allocation failed." << std::endl;
        // Handle the error appropriately.
        // Remember to deallocate any previously allocated memory.
        cudaFree(devSystemState);
        cudaFree(devConstants);
        cudaFree(devRayStatus);
        return 1;
    }

    // Allocate memory for input and output data on the device.
    // Use cudaMalloc to allocate memory for devInitCond, devConstants, devData, devStatus,
    // devDiskTexture, devSphereTexture, and devImage.

    // Call the setInitialConditions kernel to compute initial conditions and constants.
    std::cout << "Calling setInitialConditions kernel..." << std::endl;
    dim3 blockDimInit(16, 16);
    dim3 gridDimInit((IMG_COLS + blockDimInit.x - 1) / blockDimInit.x, (IMG_ROWS + blockDimInit.y - 1) / blockDimInit.y);
    setInitialConditions<<<gridDimInit, blockDimInit>>>(camera, kerrConstants, devSystemState,devConstants, pixelWidth, pixelHeight);


    // Call your main simulation kernel.
    std::cout << "Calling kernel..." << std::endl;
    dim3 blockDimKernel(16, 16);
    dim3 gridDimKernel((IMG_COLS + blockDimKernel.x - 1) / blockDimKernel.x, (IMG_ROWS + blockDimKernel.y - 1) / blockDimKernel.y);
    kernel<<<gridDimKernel, blockDimKernel>>>(kerrConstants, X0, XEND, devSystemState, H, HMAX, devConstants, devRayStatus);

    // Call the generate_image kernel to generate the final image.
    Real* devImage = nullptr;
    cudaMalloc((void**)&devImage, IMG_ROWS * IMG_COLS * 3 * sizeof(Real));

    std::cout << "Calling generate_image kernel..." << std::endl;
    dim3 blockDimGenImage(16, 16);
    dim3 gridDimGenImage((IMG_COLS + blockDimGenImage.x - 1) / blockDimGenImage.x, (IMG_ROWS + blockDimGenImage.y - 1) / blockDimGenImage.y);
    generate_image<<<gridDimGenImage, blockDimGenImage>>>(devSystemState, devRayStatus, devImage);

    // Copy the final image data from the device to the host (devImage to hostImage).
    std::cout << "Copying image data from device to host..." << std::endl;
    Real* hostImage = new Real[IMG_ROWS * IMG_COLS * 3];
    cudaMemcpy(hostImage, devImage, IMG_ROWS * IMG_COLS * 3 * sizeof(Real), cudaMemcpyDeviceToHost);

    // InitialConditions* hostInitCond = new InitialConditions[NUM_PIXELS];
    // cudaMemcpy(hostInitCond, devSystemState, NUM_PIXELS * sizeof(InitialConditions), cudaMemcpyDeviceToHost);
    // Real* hostRComponent = new Real[NUM_PIXELS];
    // for (int i = 0; i < NUM_PIXELS; i++) {
    //     hostRComponent[i] = hostInitCond[i].r; 
    // }
    // writeRawData("output.raw", hostRComponent, IMG_ROWS, IMG_COLS);

    writeRawData("output.raw", hostImage, IMG_ROWS, IMG_COLS, 3);

    // Deallocate memory on the device when done
    cudaFree(devSystemState);
    cudaFree(devConstants);
    cudaFree(devRayStatus);
    cudaFree(devImage);
    delete[] hostImage;



    return 0;
}
