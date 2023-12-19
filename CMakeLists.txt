cmake_minimum_required(VERSION 3.20.0)
project(parallel_cpp_example LANGUAGES CUDA CXX)
find_package(CUDA 11.0 REQUIRED)
add_executable(${PROJECT_NAME}_stdpar src/main.cpp)

target_compile_options(${PROJECT_NAME}_stdpar PRIVATE -stdpar)
target_link_options(${PROJECT_NAME}_stdpar PRIVATE -stdpar)

add_executable(${PROJECT_NAME} src/main.cpp)


