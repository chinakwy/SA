%module intersection_detector

%{
    #define SWIG_FILE_WITH_INIT
    #include "intersection_detector.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}


%apply (unsigned char* IN_ARRAY1, int DIM1) {(const unsigned char* p_map, int size_of_map)}
%apply (float* IN_ARRAY1, int DIM1) {(const float *ray_start, int size_ray_start)}
%apply (float* IN_ARRAY1, int DIM1) {(const float *ray_end, int size_ray_end)}

%apply (unsigned char* INPLACE_ARRAY1, int DIM1) {(unsigned char* p_isec, int size_isec)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float *p_range, int size_range)}



%include "intersection_detector.h"

