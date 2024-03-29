cmake_minimum_required(VERSION 3.20.0)
project(parallel_cpp_example LANGUAGES CXX)
############################
# Compile for GPU
############################
# make sure to select nvcc as compiler with 
# source /etc/profiles.d/module.sh
# module load ..

find_package(OpenCV REQUIRED)
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

find_package(NVHPC)
if(NVHPC_FOUND)
    target_compile_options(photometric_error PRIVATE -stdpar=multicore)
    target_link_options(photometric_error PRIVATE -stdpar=multicore)

    target_compile_options(photometric_error PRIVATE -Minfo -fast -march=native -Mllvm-fast)
    target_link_options(photometric_error PRIVATE -Minfo -fast -march=native -Mllvm-fast)

    add_executable(photometric_error_gpu src/photometric_error.cpp)
    target_include_directories(photometric_error_gpu PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(photometric_error_gpu ${OpenCV_LIBRARIES})
    target_link_libraries(photometric_error_gpu Eigen3::Eigen)

    target_compile_definitions(photometric_error_gpu PRIVATE RESOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}/resource")
    target_compile_features(photometric_error_gpu PUBLIC cxx_std_20)

    target_compile_options(photometric_error_gpu PRIVATE -stdpar=gpu)
    target_compile_options(photometric_error_gpu PRIVATE -Minfo -fast -march=native -Mllvm-fast)
    target_link_options(photometric_error_gpu PRIVATE -Minfo -fast -march=native -Mllvm-fast)
    target_link_options(photometric_error_gpu PRIVATE -stdpar=gpu)

else()
    message(WARNING "NVHPC SDK not found")
    find_package(TBB)
    if(TBB_FOUND)
        target_link_libraries(photometric_error TBB::tbb)
    endif()
endif()
