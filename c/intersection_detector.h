// i'm intersection_detector.h

#ifndef _INTERSECTION_DETECTOR_
#define _INTERSECTION_DETECTOR_

#include "stdio.h"
#include "stdbool.h"

// TODO: incorrect results while using inv_sqrt
// #define FAST_SQRT

#ifndef FAST_SQRT
#include "math.h"
#endif

#define OBSTACLE_VALUE (255) // <= 255 //which value in the map should be treated as obstacle

void do_detect(const unsigned char *p_map, int size_of_map,
             const int map_height,  const int map_width,
             const float *ray_start, int size_ray_start,
             const float *ray_end, int size_ray_end,
             unsigned char *p_isec, int size_isec,
             float *p_range, int size_range);

#endif /* _INTERSECTION_DETECTOR_ */

