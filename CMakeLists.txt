cmake_minimum_required(VERSION 3.10)
project(FastCodeAnn)

set(CMAKE_CXX_STANDARD 11)

include_directories(.)

add_executable(FastCodeAnn
        ann.cpp
        load_data.h
        network.h
        neuron.h
        sigmoid.h)
