cmake_minimum_required(VERSION 3.10)
project( OMP_histogram_equalization )

link_directories("/usr/local/Cellar/llvm/6.0.0/lib")
add_executable( OMP_histogram_equalization main.cpp )

find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( OMP_histogram_equalization ${OpenCV_LIBS} )

set(CMAKE_CXX_COMPILER "/usr/local/Cellar/llvm/6.0.0/bin/clang++")
# try also to compile and execute without: -fopenmp
set(CMAKE_CXX_FLAGS "-I/usr/local/Cellar/llvm/6.0.0/include -fopenmp")


set(SOURCE_FILES main.cpp)



