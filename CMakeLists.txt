cmake_minimum_required(VERSION 3.12)
project(RayTracingProject)

# Enable CUDA support
enable_language(CUDA)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})


# Set the source files
set(SOURCES
    functions.cu
    image_transformation.cu
    main.cu
    raytracer.cu
    solver.cu
)

# Specify the target
add_executable(raytracer ${SOURCES})
target_include_directories(raytracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(raytracer ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES})


# Set the CUDA architecture for the target
set_target_properties(raytracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET raytracer PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_options(raytracer PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>")
target_compile_options(raytracer PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
target_compile_options(raytracer PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")
target_compile_options(raytracer PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3>")