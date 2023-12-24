cmake_minimum_required(VERSION 3.20.0)
project(parallel_cpp_example LANGUAGES CXX)
# make sure to select nvcc as compiler with
# export CXX=/opt/nvidia/hpc_sdk/Linux_aarch64/21.11/compilers/bin/nvc++
# export CC=/opt/nvidia/hpc_sdk/Linux_aarch64/21.11/compilers/bin/nvc
find_package(CUDA 11.0 REQUIRED)
add_executable(${PROJECT_NAME}_stdpar src/main.cpp)

target_compile_options(${PROJECT_NAME}_stdpar PRIVATE -stdpar)
target_link_options(${PROJECT_NAME}_stdpar PRIVATE -stdpar)

add_executable(${PROJECT_NAME} src/main.cpp)


#add_library(compute_photometric_error STATIC src/compute_photometric_error.cpp)
#target_compile_options(compute_photometric_error PRIVATE -stdpar)
#target_link_options(compute_photometric_error PRIVATE -stdpar)
#target_compile_features(compute_photometric_error PUBLIC cxx_std_20)


add_executable(photometric_error src/photometric_error.cpp src/compute_photometric_error.cpp)
# Try for OpenCV 4.X, but settle for whatever is installed
find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
  find_package(OpenCV REQUIRED)
endif()
find_package(Eigen3 3.4 REQUIRED)
target_link_libraries(photometric_error Eigen3::Eigen)

message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
target_include_directories(photometric_error PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(photometric_error ${OpenCV_LIBRARIES})
target_compile_definitions(photometric_error PRIVATE RESOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}/resource")
target_compile_features(photometric_error PUBLIC cxx_std_20)
target_compile_options(photometric_error PRIVATE -stdpar=gpu)#gpu or multicore
target_link_options(photometric_error PRIVATE -stdpar=gpu)

#target_link_libraries(photometric_error compute_photometric_error)