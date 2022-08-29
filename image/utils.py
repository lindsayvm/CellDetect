import numpy as np
# import cupy as cp
# import cv2
import matplotlib.pyplot as plt
from PIL import Image


def show(i):
    i = np.asarray(i, np.float)
    mi, ma = i.min(), i.max()
    Image.fromarray(np.asarray((i - mi) / (ma - mi + 0.000001) * 255, np.uint8)).show()


def show_inline(i):
    i = np.asarray(i, np.float)
    mi, ma = i.min(), i.max()
    plt.figure()
    plt.imshow(np.asarray((i - mi) / (ma - mi + 0.000001) * 255, np.uint8))


def pad_zeros(image, num_zeros, axis, center=True):
    if num_zeros < 1:
        return image
    if axis > 1:
        raise ValueError('axis > 1')
    num_zeros_1 = num_zeros // 2
    num_zeros_2 = num_zeros_1 + (0 if (num_zeros % 2 == 0) else 1)
    shape1 = list(image.shape)
    shape2 = list(image.shape)
    shape1[axis] = num_zeros_1
    shape2[axis] = num_zeros_2
    if center:
        return np.concatenate((np.zeros(shape1), image, np.zeros(shape2)), axis=axis)
    else:
        return np.concatenate((image, np.zeros(shape1), np.zeros(shape2)), axis=axis)


def cut_tiles(image, tile_size, stride, center=True, channel_first=False):
    for h in range(0, max(1, image.shape[0] - tile_size[0] + stride[0]), stride[0]):
        for w in range(0, max(1, image.shape[1] - tile_size[1] + stride[1]), stride[1]):
            img = image[h:h + tile_size[0], w:w + tile_size[1]]
            imgp = pad_zeros(pad_zeros(img, max(tile_size[0] - img.shape[0], 0), axis=0, center=center),
                             max(tile_size[1] - img.shape[1], 0), axis=1, center=center)
            if channel_first:
                imgp = imgp.transpose((2, 0, 1))
            yield imgp


def get_matrix_tiling_positions(image_shape, tile_size, stride):
    return [((h, w), (h + tile_size[0], w + tile_size[1]))
            for w in range(0, max(1, image_shape[1] - tile_size[1] + stride[1]), stride[1]) 
            for h in range(0, max(1, image_shape[0] - tile_size[0] + stride[0]), stride[0])]


def normalize(img):
    return img / 128. - 1.


def csbdeep_normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    #if dtype is not None:
    #x   = x.astype(dtype,copy=False)
    #mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
    #ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
    
    x = (x - mi) / ( ma - mi + eps )
    #if clip:
    #    x = np.clip(x,0,1)
    return x


def overlap_2d_arrays_with_global_positions(global_top_left_1, global_bottom_right_1, global_top_left_2, global_bottom_right_2, arguments_are_lists=False):
    if arguments_are_lists:
        global_top_left_1 = np.array(global_top_left_1)
        global_bottom_right_1 = np.array(global_bottom_right_1)
        global_top_left_2 = np.array(global_top_left_2)
        global_bottom_right_2 = np.array(global_bottom_right_2)
    
    size_1 = global_bottom_right_1 - global_top_left_1
    size_2 = global_bottom_right_2 - global_top_left_2
    
    top_left_1 = np.array(([0, 0], global_top_left_2 - global_top_left_1)).max(axis=0)
    bottom_right_1 = np.array((size_1, global_bottom_right_2 - global_top_left_1)).min(axis=0)
    bottom_right_1 = np.array((bottom_right_1, top_left_1)).max(axis=0)

    top_left_2 = np.array(([0, 0], global_top_left_1 - global_top_left_2)).max(axis=0)
    bottom_right_2 = np.array((size_2, global_bottom_right_1 - global_top_left_2)).min(axis=0)
    bottom_right_2 = np.array((bottom_right_2, top_left_2)).max(axis=0)
    
    return top_left_1, bottom_right_1, top_left_2, bottom_right_2


def overlap_2d_arrays_with_point(array1_shape, array2_shape, overlay1, overlay2, arguments_are_lists=False):
    
    if arguments_are_lists:
        array1_shape = np.array(array1_shape)
        array2_shape = np.array(array2_shape)
        overlay1 = np.array(overlay1)
        overlay2 = np.array(overlay2)
        
    o1_o2 = overlay1 - overlay2
    a1_a2 = array1_shape - array2_shape
    top_left_1 = (o1_o2 > 0) * o1_o2
    top_left_2 = (o1_o2 < 0) * -o1_o2
    bottom_right_1 = (a1_a2 < o1_o2) * (a1_a2 - o1_o2) + o1_o2 + array2_shape 
    bottom_right_2 = (a1_a2 > o1_o2) * (o1_o2 - a1_a2) - o1_o2 + array1_shape

    return top_left_1, bottom_right_1, top_left_2, bottom_right_2
