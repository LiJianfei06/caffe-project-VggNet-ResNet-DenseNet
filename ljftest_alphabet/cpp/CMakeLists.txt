cmake_minimum_required (VERSION 2.8)  
  
SET (SRC_LIST classification_so.cpp)  
include_directories ( /home/ljf/caffe-master/include  
    /usr/local/include  
    /usr/local/cuda/include  
    /usr/include  
    /home/ljf/caffe-master/cmake )  
  
add_library(classification_so SHARED ${SRC_LIST})
