#include <cassert>
#include <fstream>
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "common.h"
#include "image_transformation.cuh"
#include "raytracer.cuh"

BlackHoleConstants
calculateKerrConstants(Real a, const Camera& camera)
{
    BlackHoleConstants constants{};
    constants.a = a;
    constants.a2 = a * a;

    Real const r2 = camera.r * camera.r;
    Real const a2 = a * a;
    // Calculate the constants
    constants.ro = sqrt(r2 + a2 * cos(camera.theta) * cos(camera.theta));
    constants.delta = r2 + a2;
    constants.sigma =
            sqrt((r2 + a2) * (r2 + a2) - a2 * constants.delta * sin(camera.theta) * sin(camera.theta));
    constants.alpha = constants.ro * sqrt(constants.delta) / constants.sigma;
    constants.omega = 2.0 * a * camera.r / (constants.sigma * constants.sigma);
    constants.pomega = constants.sigma * sin(camera.theta) / constants.ro;
    return constants;
}

class BlackHoleSimulator
{
  public:
    BlackHoleSimulator()
    {
        assert(instance == nullptr);
        instance = this;
    }

    void Initialize()
    {
        // Initialize OpenGL and create a window
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
        glutInitWindowSize(IMG_COLS, IMG_ROWS);
        glutCreateWindow("Black Hole Simulation");

        // Initialize GLEW
        GLenum const err = glewInit();
        if (err != GLEW_OK) {
            std::cerr << "GLEW initialization error: " << glewGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Set up OpenGL context and callbacks
        glutDisplayFunc(DisplayCallbackWrapper);
        glutKeyboardFunc(KeyboardCallbackWrapper);

        // Initialize CUDA
        cudaSetDevice(0);  // Select the CUDA device (change as needed).

        // Allocate memory for input and output data on the host.
        cudaMalloc((void**)&devSystemState, NUM_PIXELS * SYSTEM_SIZE * sizeof(Real));
        cudaMalloc((void**)&devConstants, NUM_PIXELS * 2 * sizeof(Real));
        cudaMalloc((void**)&devRayStatus, NUM_PIXELS * sizeof(int));

        // Check for CUDA allocation errors (optional)
        if (devSystemState == nullptr || devConstants == nullptr || devRayStatus == nullptr) {
            std::cerr << "CUDA allocation failed." << std::endl;
            cudaFree(devSystemState);
            cudaFree(devConstants);
            cudaFree(devRayStatus);
            exit(1);
        }

        // Initialize CUDA-OpenGL interoperability
        glGenBuffers(1, &cudaGLBuffer);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, cudaGLBuffer);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, NUM_PIXELS * 3 * sizeof(Real), nullptr, GL_DYNAMIC_DRAW);
        cudaGraphicsGLRegisterBuffer(&cudaGLResource, cudaGLBuffer, cudaGraphicsMapFlagsNone);
    }

    static void Run()
    {
        // Start the OpenGL main loop
        glutMainLoop();
    }

    static void DisplayCallbackWrapper()
    {
        if (instance) {
            instance->display();
        }
    }

    static void KeyboardCallbackWrapper(unsigned char key, int x, int y)
    {
        if (instance) {
            instance->keyboard(key, x, y);
        }
    }

    ~BlackHoleSimulator()
    {
        cudaGraphicsUnregisterResource(cudaGLResource);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, cudaGLBuffer);
        glDeleteBuffers(1, &cudaGLBuffer);
        glutDestroyWindow(glutGetWindow());

        // Deallocate memory on the device when done
        cudaFree(devSystemState);
        cudaFree(devConstants);
        cudaFree(devRayStatus);

        instance = nullptr;
    }

  private:
    int argc = 0;
    char** argv = nullptr;

    cudaGraphicsResource* cudaGLResource{nullptr};
    GLuint cudaGLBuffer{};
    Real* devSystemState{nullptr};
    Real* devConstants{nullptr};
    int* devRayStatus{nullptr};

    const Real CAM_SENSOR_HEIGHT = 16.0;
    const Real CAM_SENSOR_WIDTH = 16.0;

    const Real H = -0.001;
    const Real HMAX = -150;

    const Real X0 = 0;
    const Real XEND = -150;

    Real a{0.2647333333333333};
    Real r{40.0};
    Real theta{1.413717694115407};
    Real phi{0.0};

    Real roll{0.0};
    Real pitch{0.0};
    Real yaw{-0.06};

    static BlackHoleSimulator* instance;

    void display()
    {
        // Compute the width and height of a pixel
        Real const pixelWidth = CAM_SENSOR_WIDTH / static_cast<Real>(IMG_COLS);
        Real const pixelHeight = CAM_SENSOR_HEIGHT / static_cast<Real>(IMG_ROWS);

        // Initialize the black hole constants
        Camera const camera = {r, theta, phi, roll, pitch, yaw, r / 3, 0.0, pixelWidth, pixelHeight};
        BlackHoleConstants const kerrConstants = calculateKerrConstants(a, camera);

        cudaMemset(devRayStatus, 0, NUM_PIXELS * sizeof(int));
        dim3 const blockDimKernel(16, 16);
        dim3 const gridDimKernel(
                (IMG_COLS + blockDimKernel.x - 1) / blockDimKernel.x,
                (IMG_ROWS + blockDimKernel.y - 1) / blockDimKernel.y);
        raytrace<<<gridDimKernel, blockDimKernel>>>(
                camera,
                kerrConstants,
                X0,
                XEND,
                devSystemState,
                H,
                HMAX,
                devConstants,
                devRayStatus);

        std::cout << "Calling generate_image kernel..." << std::endl;
        // Map the CUDA buffer to OpenGL
        float* d_mapped_buffer = nullptr;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &cudaGLResource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_mapped_buffer, &num_bytes, cudaGLResource);

        std::cout << "Generate final image..." << std::endl;
        // Call the generate_image kernel to generate the final image.
        dim3 const blockDimGenImage(16, 16);
        dim3 const gridDimGenImage(
                (IMG_COLS + blockDimGenImage.x - 1) / blockDimGenImage.x,
                (IMG_ROWS + blockDimGenImage.y - 1) / blockDimGenImage.y);
        generate_image<<<gridDimGenImage, blockDimGenImage>>>(
                devSystemState,
                devRayStatus,
                d_mapped_buffer);

        // Unmap the CUDA buffer from OpenGL
        cudaGraphicsUnmapResources(1, &cudaGLResource, 0);

        glClear(GL_COLOR_BUFFER_BIT);

        // If a non-zero named buffer object is bound to the
        // GL_PIXEL_UNPACK_BUFFER target (see main function) while a block of
        // pixels is specified, data is treated as a byte offset into the buffer
        // object's data store.
        glDrawPixels(IMG_COLS, IMG_ROWS, GL_RGB, GL_FLOAT, 0);  // Use GL_RGB format

        glutSwapBuffers();
    }

    void keyboard(unsigned char key, int x, int y)
    {
        switch (key) {
            case 'a':  // Move r closer to the black hole
                r -= 1.0;
                break;
            case 'd':  // Move r away from the black hole
                r += 1.0;
                break;
            case 'w':  // Increase theta (look up)
                theta += 0.01;
                break;
            case 's':  // Decrease theta (look down)
                theta -= 0.01;
                break;
            case 'q':  // Rotate phi counterclockwise
                phi -= 0.01;
                break;
            case 'e':  // Rotate phi clockwise
                phi += 0.01;
                break;
            case 'o':
                a -= 0.01;
                break;
            case 'p':
                a += 0.01;
                break;
            case 27:  // ESC key to exit
                exit(0);
                break;
        }
        glutPostRedisplay();
    }
};

BlackHoleSimulator* BlackHoleSimulator::instance = nullptr;

int
main(int argc, char** argv)
{
    BlackHoleSimulator simulator;
    simulator.Initialize();
    simulator.Run();
    return 0;
}