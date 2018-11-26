import sys
sys.path.append('./build/lib.macosx-10.6-x86_64-3.5/')

import intersection_detector
import numpy as np
import matplotlib.pyplot as plt

#genrate a map
height = 7
width = 9
map = np.zeros(height*width,dtype=np.uint8)
for i in range(height):
    for j in range(width):
        if ((i <= 1) or (i >= height - 2)or(j <= 1) or (j >= width - 2)):
            map[i*width + j] = 255
        else:
            map[i*width + j] = 0

map_img = map.reshape((height,width))
plt.imshow(map_img)

print("Map:\n",end='')
print("            ",end='')
for i in range(height):
    for j in range(width):
        if (map[i*width + j] == 0):
            print("  ",end='') #free
        else:
            print("* ",end='') #obstacle
    print("\n",end='')
    print("            ",end='')
print("\n",end='')

# setting rays
ray_start = np.array([4,4,4,4,4,4,4,4,
                      3,3,3,3,3,3,3,3],dtype=np.float32)

ray_end   = np.array([7,7,4,1,1,1,4,7,
                      3,1,1,1,3,5,5,5],dtype=np.float32)

N_rays = (int)(ray_start.size / 2)

# outputs
isect = np.zeros(N_rays,dtype=np.uint8)
ray_range = np.zeros(N_rays,dtype=np.float32)

# compute intersection
intersection_detector.do_detect(map,height,width,ray_start,ray_end,isect,ray_range)

# show results
for i in range(N_rays):
    print('Ray:%d \t Start point: [%5.2f,%5.2f]\t End point: [%5.2f, %5.2f]'%(i,ray_start[i],ray_start[i+N_rays],ray_end[i],ray_end[i+N_rays]))
print('\n')
for i in range(N_rays):
    print('Ray:%d \t intersect:%s \t range:%5.3f' % (i,"true" if isect[i]==1 else "false",ray_range[i]))

# plt.show()




