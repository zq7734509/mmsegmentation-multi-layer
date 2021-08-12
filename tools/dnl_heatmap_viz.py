import torch.tensor
import numpy as np
from PIL import Image
import cv2

##      -----> x
##      |
##      |
##      ↓ y
## 顺序(x,y)
def dnl_weight_func(pair_array, unary_array, img, pixel_x, pixel_y, str, ind): # 先x后y
    w, h = img.size
    scale = 4

    w = w // scale
    h = h // scale
    pixel_x = pixel_x // scale
    pixel_y = pixel_y // scale

    pos = pixel_x + pixel_y * w  # 按行填满

    unary_max = unary_array.max()
    unary_min = unary_array.min()
    pair_max = pair_array[pos, :].max()
    pair_min = pair_array[pos, :].min()

    scale = (unary_max - unary_min) / (pair_max - pair_min) * 1

    pos_weight = pair_array[pos, :] * scale + unary_array

    index = np.argsort(pos_weight)
    smallest_indexes = index[:50]
    min_val = pos_weight[index[50]]
    biggest_indexes = index[-50:]
    max_val = pos_weight[index[-50]]

    for i in smallest_indexes:
        pos_weight[i] = min_val
    for i in biggest_indexes:
        pos_weight[i] = max_val

    # max_val = pos_weight.max()

    pos_weight = (pos_weight - min_val) / (max_val - min_val)
    pos_weight = pos_weight.reshape(h, w)
    # pos_weight_mat = np.mat(pos_weight)
    # img_output = np.uint8(255 * pos_weight_mat)
    # cv2.applyColorMap(img_output, cv2.COLORMAP_JET)
    # cv2.imshow('window',img_output)

    img_output = np.uint8(255 * pos_weight)
    img_color = cv2.applyColorMap(img_output, cv2.COLORMAP_JET)
    img_color_resize = cv2.resize(img_color, (300, 300))
    output_str = '/home/allen/mmseg_viz/dnl_map/' + ind + '_' + str + '_dnl.png'
    cv2.imwrite(output_str, img_color_resize)

##      -----> x
##      |
##      |
##      ↓ y
## 顺序(x,y)
def pairwise_weight_func(array, img, pixel_x, pixel_y, str, ind): # 先x后y
    w, h = img.size
    scale = 4

    w = w // scale
    h = h // scale
    pixel_x = pixel_x // scale
    pixel_y = pixel_y // scale

    pos = pixel_x + pixel_y * w  # 按行填满
    pos_weight = array[pos,:]

    index = np.argsort(pos_weight)
    smallest_indexes = index[:50]
    min_val = pos_weight[index[50]]
    biggest_indexes = index[-50:]
    max_val = pos_weight[index[-50]]

    for i in smallest_indexes:
        pos_weight[i] = min_val
    for i in biggest_indexes:
        pos_weight[i] = max_val

    # max_val = pos_weight.max()

    pos_weight = (pos_weight-min_val)/(max_val-min_val)
    pos_weight = pos_weight.reshape(h,w)
    # pos_weight_mat = np.mat(pos_weight)
    # img_output = np.uint8(255 * pos_weight_mat)
    # cv2.applyColorMap(img_output, cv2.COLORMAP_JET)
    # cv2.imshow('window',img_output)

    img_output = np.uint8(255 * pos_weight)
    img_color = cv2.applyColorMap(img_output, cv2.COLORMAP_JET)
    img_color_resize = cv2.resize(img_color,(300,300))
    output_str = '/home/allen/mmseg_viz/pairwise_map/' + ind + '_' + str + '_pair.png'
    cv2.imwrite(output_str,img_color_resize)

    # status = np.zeros((w*h,2))
    # for r in range(w*h):
    #     status[r][0] = array[r,:].min()
    #     status[r][1] = array[r,:].max()

def unary_mask_func(array, img, str): # 先x后y
    w, h = img.size
    scale = 4

    w = w // scale
    h = h // scale

    unary_mask = array
    index = np.argsort(unary_mask)
    smallest_indexes = index[:50]
    min_val = unary_mask[index[50]]

    for i in smallest_indexes:
        unary_mask[i] = min_val

    max_val = unary_mask.max()

    unary_mask = (unary_mask - min_val) / (max_val - min_val)
    unary_mask = unary_mask.reshape(h, w)

    img_output = np.uint8(255 * unary_mask)
    img_color = cv2.applyColorMap(img_output, cv2.COLORMAP_HOT)
    img_color_resize = cv2.resize(img_color, (300, 300))
    output_str = '/home/allen/mmseg_viz/unary_map/' + str + '_unary.png'
    cv2.imwrite(output_str, img_color_resize)

file_index = 1

img_file_path = '/home/allen/mmseg_viz/dnl_origin_img/' + str(file_index).zfill(6) + '.png'
unary_mask_file_path = '/home/allen/mmseg_viz/unary_mask/' + str(file_index).zfill(6) + '.txt'
pairwise_weight_file_path = '/home/allen/mmseg_viz/pairwise_weight/' + str(file_index).zfill(6) + '.txt'

unary_mask_array = np.loadtxt(unary_mask_file_path)
pairwise_weight_array = np.loadtxt(pairwise_weight_file_path)
img = Image.open(img_file_path)

##      o -----> x
##      |
##      |
##      ↓ y
## 顺序(x,y)
# if file_index==3:
#     pairwise_weight_func(pairwise_weight_array, img, 189, 141, 'car', str(file_index))
#     pairwise_weight_func(pairwise_weight_array, img, 262, 85, 'build', str(file_index))
#     pairwise_weight_func(pairwise_weight_array, img, 196, 190, 'grass', str(file_index))
#     pairwise_weight_func(pairwise_weight_array, img, 45, 34, 'road', str(file_index))
#     unary_mask_func(unary_mask_array, img, str(file_index))
#     dnl_weight_func(pairwise_weight_array, unary_mask_array, img, 189, 141, 'car', str(file_index))
#     dnl_weight_func(pairwise_weight_array, unary_mask_array, img, 262, 85, 'build', str(file_index))
#     dnl_weight_func(pairwise_weight_array, unary_mask_array, img, 196, 190, 'grass', str(file_index))
#     dnl_weight_func(pairwise_weight_array, unary_mask_array, img, 45, 34, 'road', str(file_index))
#
# if file_index==14:
#     pairwise_weight_func(pairwise_weight_array, img, 223, 56, 'car', str(file_index))
#     pairwise_weight_func(pairwise_weight_array, img, 260, 249, 'build', str(file_index))
#     pairwise_weight_func(pairwise_weight_array, img, 43, 245, 'grass', str(file_index))
#     pairwise_weight_func(pairwise_weight_array, img, 45, 46, 'road', str(file_index))
#     unary_mask_func(unary_mask_array, img, str(file_index))
#     dnl_weight_func(pairwise_weight_array, unary_mask_array, img, 223, 56, 'car', str(file_index))
#     dnl_weight_func(pairwise_weight_array, unary_mask_array, img, 260, 249, 'build', str(file_index))
#     dnl_weight_func(pairwise_weight_array, unary_mask_array, img, 43, 245, 'grass', str(file_index))
#     dnl_weight_func(pairwise_weight_array, unary_mask_array, img, 45, 46, 'road', str(file_index))
#
# if file_index==11:
#     pairwise_weight_func(pairwise_weight_array, img, 274, 115, 'car', str(file_index))
#     pairwise_weight_func(pairwise_weight_array, img, 259, 246, 'build', str(file_index))
#     pairwise_weight_func(pairwise_weight_array, img, 151, 172, 'grass', str(file_index))
#     pairwise_weight_func(pairwise_weight_array, img, 153, 53, 'road', str(file_index))
#     unary_mask_func(unary_mask_array, img, str(file_index))
#     dnl_weight_func(pairwise_weight_array, unary_mask_array, img, 274, 115, 'car', str(file_index))
#     dnl_weight_func(pairwise_weight_array, unary_mask_array, img, 259, 246, 'build', str(file_index))
#     dnl_weight_func(pairwise_weight_array, unary_mask_array, img, 151, 172, 'grass', str(file_index))
#     dnl_weight_func(pairwise_weight_array, unary_mask_array, img, 153, 53, 'road', str(file_index))

interval = 5
w, h = img.size
for i in range(0, h, interval):
    for j in range(0, w, interval):
        pos_x = i
        pos_y = j
        name = str(i) + '_' + str(j)
        pairwise_weight_func(pairwise_weight_array, img, pos_x, pos_y, name, str(file_index))
        dnl_weight_func(pairwise_weight_array, unary_mask_array, img, pos_x, pos_y, name, str(file_index))
unary_mask_func(unary_mask_array, img, str(file_index))