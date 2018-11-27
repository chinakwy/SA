import numpy as np
import sys
sys.path.append('./c/build/lib.macosx-10.6-x86_64-3.5/')
sys.path.append('./c/build/lib.macosx-10.12-x86_64-3.7/')
sys.path.append('./c/build/lib.linux-x86_64-3.5/')
from c import intersection_detector


class ISectGridMapRays:
    def __init__(self):
        self.height = 600
        self.width = 800

    def intersection_gridmap_rays(self, map_obstacles, raystart, rayend):
        height = self.height
        width = self.width
        n_rays = len(raystart)
        map_ = np.array(map_obstacles.reshape(-1), dtype=np.uint8)
        ray_start = np.array(np.concatenate((raystart[:, 0], raystart[:, 1]), axis=None), dtype=np.float32)
        ray_end = np.array(np.concatenate((rayend[:, 0], rayend[:, 1]), axis=None), dtype=np.float32)
        isect = np.zeros(n_rays, dtype=np.uint8)
        ray_range = np.zeros(n_rays, dtype=np.float32)

        intersection_detector.do_detect(map_, height, width, ray_start, ray_end, isect, ray_range)

        return isect, ray_range
