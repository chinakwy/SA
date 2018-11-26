// i'm intersection_detector.c

#include "intersection_detector.h"

struct pPoint
{
  float *pX;
  float *pY;
};

float inv_sqrt(float number)
{
  long i;
  float x2, y;
  const float threehalfs = 1.5F;

  x2 = number * 0.5F;
  y = number;
  i = *(long *)&y;           // evil floating point bit level hacking
  i = 0x5f3759df - (i >> 1); // what the fuck?
  y = *(float *)&i;
  y = y * (threehalfs - (x2 * y * y)); // 1st iteration
  y = y * (threehalfs - (x2 * y * y)); // 2nd iteration

  return y;
}

/**
 * test rays for intersection with an obstacle grid map and 
 * return the distance from the start point to the first obstacle.
 * 
 * Inputs:
 * - p_map: grayscale map for obstacle value (0~255), 1-D array(uint8)
 * - map_height: height of the map, int
 * - map_width: width of the map, int
 * - ray_start: start points of the rays, 1-D array(float32)
 * - ray_end: end points of the rays, 1-D array(float32)
 * 
 * Outputs: 
 * - p_isec: =1 if the ray intersects an obstacle else=0, 1-D array(uint8)
 * - p_range: distance from start to first obstacle or (-1) when no intersection was found,
 *            1-D array(float32)
 * 
 * The map: 
 * 1-D Array, start from upper left, end at lower right
 * 
 * (0,0) +-----------> X (width-1,0)
 * (0,1) |  --------->   (width-1,1)
 *       :  _________|
 *       : |
 *       | ---------->
 *     Y v            (width-1,height-1)
*/
void do_detect(const unsigned char *p_map, int size_of_map,
             const int map_height,  const int map_width,
             const float *ray_start, int size_ray_start,
             const float *ray_end, int size_ray_end,
             unsigned char *p_isec, int size_isec,
             float *p_range, int size_range)
{
  struct pPoint rayStart, rayEnd;
  const unsigned char* pMap = p_map;
  int height = map_height;
  int width = map_width;
  int pitch = width;
  //=========================================================================================
  unsigned char obstacleValue = OBSTACLE_VALUE; //which value in the map should be treated as obstacle

  int count = size_ray_start/2;        // number fo rays
  rayStart.pX = (float *)ray_start;
  rayStart.pY = rayStart.pX + count;
  rayEnd.pX = (float *)ray_end;
  rayEnd.pY = rayEnd.pX + count;
  
  unsigned char *pISect = p_isec;
  float *pRange = p_range;

  for (int i = 0; i < count; i++)
  {
    int xs = (int)((*rayStart.pX++)); /* add rounding constant of 0.5 */
    int ys = (int)((*rayStart.pY++));
    if (xs >= 0 && ys >= 0 && xs < width && ys < height)
    {
      if (pMap[xs + ys*pitch] != obstacleValue)
      {
        int xe = (int)((*rayEnd.pX++));
        int ye = (int)((*rayEnd.pY++));

        int dx = xe - xs;
        int dy = ye - ys;
        int x, y, prevX, prevY;
        bool isect;

#define CHECK_CELL                          \
  if (pMap[x + y*pitch] != obstacleValue) \
  {                                         \
    prevX = x;                              \
    prevY = y;                              \
  }                                         \
  else                                      \
  {                                         \
    break;                                  \
  }

        if (dx >= 0)
        { /* octants 1, 2, 7, 8 */
          if (dy >= 0)
          {               /* octants 1, 2 */
            if (dx >= dy) //Bresenham algorithm
            {             /* octant 1 */
              int error = dx >> 1;
              int xe_ = xe < width ? xe : (width - 1);
              for (prevX = x = xs, prevY = y = ys; x <= xe_; x++)
              {
                CHECK_CELL
                error -= dy;
                if (error < 0)
                {
                  if (++y >= height)
                    break;
                  error += dx;
                }
              }
              isect = (x <= xe);
            }
            else
            { /* octant 2 */
              int error = dy >> 1;
              int ye_ = ye < height ? ye : (height - 1);
              for (prevX = x = xs, prevY = y = ys; y <= ye_; y++)
              {
                CHECK_CELL
                error -= dx;
                if (error < 0)
                {
                  if (++x >= width)
                    break;
                  error += dy;
                }
              }
              isect = (y <= ye);
            }
          }
          else
          { /* octants 7, 8 */
            dy = -dy;
            if (dx >= dy)
            { /* octant 8 */
              int error = dx >> 1;
              int xe_ = xe < width ? xe : (width - 1);
              for (prevX = x = xs, prevY = y = ys; x <= xe_; x++)
              {
                CHECK_CELL
                error -= dy;
                if (error < 0)
                {
                  if (--y < 0)
                    break;
                  error += dx;
                }
              }
              isect = (x <= xe);
            }
            else
            { /* octant 7 */
              int error = dy >> 1;
              int ye_ = ye < 0 ? 0 : ye;
              for (prevX = x = xs, prevY = y = ys; y >= ye_; y--)
              {
                CHECK_CELL
                error -= dx;
                if (error < 0)
                {
                  if (++x >= width)
                    break;
                  error += dy;
                }
              }
              isect = (y >= ye);
            }
          }
        }
        else
        { /* octants 3-6 */
          dx = -dx;
          if (dy >= 0)
          { /* octants 3, 4 */
            if (dx >= dy)
            { /* octant 4 */
              int error = dx >> 1;
              int xe_ = xe < 0 ? 0 : xe;
              for (prevX = x = xs, prevY = y = ys; x >= xe_; x--)
              {
                CHECK_CELL
                error -= dy;
                if (error < 0)
                {
                  if (++y >= height)
                    break;
                  error += dx;
                }
              }
              isect = (x >= xe);
            }
            else
            { /* octant 3 */
              int error = dy >> 1;
              int ye_ = ye < height ? ye : (height - 1);
              for (prevX = x = xs, prevY = y = ys; y <= ye_; y++)
              {
                CHECK_CELL
                error -= dx;
                if (error < 0)
                {
                  if (--x < 0)
                    break;
                  error += dy;
                }
              }
              isect = (y <= ye);
            }
          }
          else
          { /* octants 5, 6 */
            dy = -dy;
            if (dx >= dy)
            { /* octant 5 */
              int error = dx >> 1;
              int xe_ = xe < 0 ? 0 : xe;
              for (prevX = x = xs, prevY = y = ys; x >= xe_; x--)
              {
                CHECK_CELL
                error -= dy;
                if (error < 0)
                {
                  if (--y < 0)
                    break;
                  error += dx;
                }
              }
              isect = (x >= xe);
            }
            else
            { /* octant 6 */
              int error = dy >> 1;
              int ye_ = ye < 0 ? 0 : ye;
              for (prevX = x = xs, prevY = y = ys; y >= ye_; y--)
              {
                CHECK_CELL
                error -= dx;
                if (error < 0)
                {
                  if (--x < 0)
                    break;
                  error += dy;
                }
              }
              isect = (y >= ye);
            }
          }
        }
#undef CHECK_CELL

        *pISect++ = isect?1:0;

        #ifdef FAST_SQRT
        *pRange++ = isect ? (1.0 / inv_sqrt((prevX - xs) * (prevX - xs) + (prevY - ys) * (prevY - ys))) : -1.0;
        #else
         *pRange++ = isect ? sqrt((double)((prevX - xs) * (prevX - xs) + (prevY - ys) * (prevY - ys))) : -1.0;
        #endif
        
        continue;
      }
    }

    // ray starts off the map or the start position is an obstacle
    *pISect++ = 1;
    if (pRange)
      *pRange++ = 0.0;
  }
}

