'''注意与显示的不一样，地图上下颠倒'''

import numpy as np
import glob
from skimage import io, transform
import pandas as pd


# map = 'office.gif'  # 600*800
# map = 'emptyroom.gif'  # 600*800
map = 'map_office.gif'  # 300*400 for a_star


def read_map(map_path):
    print('read_map')
    im = glob.glob(map_path)
    img = io.imread(im[0])
    # landmarks index
    # imag_array = img[:, :, 1]  #每行每列的数组的第一个元素，形成一个数组，为什么要做这一步❓✅可能是颜色只要看第一个就好了.考虑到绿色 选第二个元素（RGB）
    imag_array = img
    imag_array_list = np.ndarray.tolist(imag_array)

    # ========================================================================白色定为0  黑色定为255
    #     for i in range(len(imag_array_list)):#column_len 柱长即几行
    #         for j in range(len(imag_array_list[0])):#row_len 行长即几列
    #             if imag_array_list[i][j] == 255:
    #                 imag_array_list[i][j] = 1  # 将地图二值化 白色路转成一
    #
    #     for i in range(len(imag_array_list)):
    #         for j in range(len(imag_array_list[0])):
    #             if imag_array_list[i][j] == 0:
    #                 # imag_array_list[i][j] = int(imag_array_list[i][j])
    #                 imag_array_list[i][j] = 255  # 将0黑色障碍转成255
    #
    #     for i in range(len(imag_array_list)):#column_len 柱长即几行
    #         for j in range(len(imag_array_list[0])):#row_len 行长即几列
    #             if imag_array_list[i][j] == 1:
    #                 imag_array_list[i][j]= 0  # 将地图二值化 白色路转成0
    # ===========================================================================

    # =======================================================================白色定为0 黑色定为1
    for i in range(len(imag_array_list)):  # column_len 柱长即几行
        for j in range(len(imag_array_list[0])):  # row_len 行长即几列
            if imag_array_list[i][j] == 0:
                imag_array_list[i][j] = 1
    for i in range(len(imag_array_list)):
        for j in range(len(imag_array_list[0])):
            if imag_array_list[i][j] == 255:
                imag_array_list[i][j] = 0


# ========================================================================白色 '.', 黑色'#'
#     for i in range(len(imag_array_list)):  # column_len 柱长即几行
#         for j in range(len(imag_array_list[0])):  # row_len 行长即几列
#             if imag_array_list[i][j] == 255:
#                 imag_array_list[i][j] = '.'
#     for i in range(len(imag_array_list)):
#         for j in range(len(imag_array_list[0])):
#             if imag_array_list[i][j] == 0:
#                 imag_array_list[i][j] = '#'
# =================================================================


# =============================================================================
imag_array_list = read_map(map)
df = pd.DataFrame(imag_array_list)
np.save("filename.npy", df)

# df.to_csv("office_255black_0white.txt",index=False,sep=",",header=False)#存一个emptyroom_imagine.txt
# =============================================================================
