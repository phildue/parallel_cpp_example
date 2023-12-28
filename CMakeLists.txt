cmake_minimum_required(VERSION 3.20.0)
project(parallel_cpp_example LANGUAGES CXX)
############################
# Compile for GPU
############################
# make sure to select nvcc as compiler with 
# source /etc/profiles.d/module.profile
# module load ..

find_package(OpenCV)
find_package(Eigen3 3.4 REQUIRED)

add_executable(photometric_error src/photometric_error.cpp)

target_include_directories(photometric_error PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(photometric_error ${OpenCV_LIBRARIES})

target_link_libraries(photometric_error Eigen3::Eigen)

target_compile_definitions(photometric_error PRIVATE RESOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}/resource")
target_compile_features(photometric_error PUBLIC cxx_std_20)

#######################################################
#   Link various parallelization backends if available
#######################################################
find_package(TBB QUIET)
if(TBB_FOUND)
    target_link_libraries(photometric_error TBB::tbb)
endif()
find_package(NVHPC QUIET)
if(NVHPC_FOUND)
    message(STATUS "Found NVHPC")
    target_compile_options(photometric_error PRIVATE -stdpar=multicore)#gpu or multicore
    target_link_options(photometric_error PRIVATE -stdpar=multicore)
    add_executable(photometric_error_gpu src/photometric_error.cpp)
    target_include_directories(photometric_error_gpu PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(photometric_error_gpu ${OpenCV_LIBRARIES})
    target_link_libraries(photometric_error_gpu Eigen3::Eigen)

    target_compile_definitions(photometric_error_gpu PRIVATE RESOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}/resource")
    target_compile_features(photometric_error_gpu PUBLIC cxx_std_17)

    target_compile_options(photometric_error_gpu PRIVATE -stdpar=gpu)#gpu or multicore
    target_link_options(photometric_error_gpu PRIVATE -stdpar=gpu)
endif()