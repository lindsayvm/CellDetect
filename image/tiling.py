import numpy as np
import cupy as cp
import itertools
import sys
from csbdeep.utils import normalize as csbdeep_normalize
from PIL import UnidentifiedImageError

sys.path.append('../../')
from image.utils import get_matrix_tiling_positions, overlap_2d_arrays_with_point, overlap_2d_arrays_with_global_positions



class Tiling:
    # assumes fixed tile size for all patches
    # assumes level is knowm and fixed
    # assumes all adjusted coordinated to match the level coordinates
    # assumes all times and patches are from the same slide
    def __init__(self, yx_patch_size, yx_tile_size, 
                 get_tiles_fn, key_image_id, key_level_value,
                 yx_locations_topleft=None, yx_locations_centers=None, 
                 cuda_device_index=None): ### , csbdeep_normalize=False
        
        self.yx_patch_size = yx_patch_size
        self.yx_tile_size = yx_tile_size
        
        
        def _d(k1, k2, k3):
            try:
                til = get_tiles_fn(k1, k2, k3)
                if til.shape == (*yx_tile_size, 3):
                    return til
                else: raise UnidentifiedImageError()
            except UnidentifiedImageError:
                print('UnidentifiedImageError', k1, k2, k3)
                return np.zeros((*yx_tile_size, 3))

        self.get_tiles_fn = _d
        
        # when key setting and also calling slidescore API
        # it is the only place where we have x first and y scecond:
        # API: /i/<imageID>/<urlPart>/i_files/<level>/<column>_<row>.jpeg
        # tile key = (image_id, (num_x, num_y), level)
        self._key_maker = lambda y, x: (key_image_id, (x, y), key_level_value)
        ### k = list(self.tiles.keys())[0]
        ### self._key_maker = lambda y, x: (k[0], (y, x), k[2])
        
        if yx_locations_topleft is not None:
            self.yx_locations_topleft = np.array(yx_locations_topleft)
        elif yx_locations_centers is not None:
            margin = np.tile((np.array(self.yx_patch_size) / 2).astype(int), (yx_locations_centers.shape[0], 1))
            self.yx_locations_topleft = np.array(yx_locations_centers) - margin
        else:
            raise ValueError('either yx_locations_topleft or yx_locations_centers should be provided.')


        yx_locations_bottomright = self.yx_locations_topleft + np.tile(self.yx_patch_size, (self.yx_locations_topleft.shape[0], 1))

        topleft_patch_tile_nums = (self.yx_locations_topleft / np.tile(self.yx_tile_size, (self.yx_locations_topleft.shape[0], 1))).astype(int)
        bottomright_path_tile_nums_excl = (yx_locations_bottomright / np.tile(self.yx_tile_size, (yx_locations_bottomright.shape[0], 1))).astype(int) + 1 # to have it excluded
        
        self.patch_tile_nums = [ # per patch tile numbers
            np.array(list(itertools.product(range(topleft_patch_tile_nums[i][0], bottomright_path_tile_nums_excl[i][0]), 
                                            range(topleft_patch_tile_nums[i][1], bottomright_path_tile_nums_excl[i][1]))))
            for i in range(topleft_patch_tile_nums.shape[0])]
        
        # bottom right will not work if the tile sizes may vary, was originally ottom_right_tile = (top_left_tile[0] + tile.shape[0], top_left_tile[1] + tile.shape[1])
        patch_tile_yx_topleft = [nums * np.tile(self.yx_tile_size, (nums.shape[0], 1)) for nums in self.patch_tile_nums]
        patch_tile_yx_bottomright = [toplefts + np.tile(self.yx_tile_size, (toplefts.shape[0], 1)) for toplefts in patch_tile_yx_topleft]
        
        
        # doing the overlaping [[(patch_from, patch_to, tile_from, tile_to) for each tile] for each patch]
        # overlap...(global_top_left_1, global_bottom_right_1, global_top_left_2, global_bottom_right_2) returns top_left_1, bottom_right_1, top_left_2, bottom_right_2
        self.patch_and_tile_cuts = [[ overlap_2d_arrays_with_global_positions(
            self.yx_locations_topleft[i], yx_locations_bottomright[i], patch_tile_yx_topleft[i][j], patch_tile_yx_bottomright[i][j], arguments_are_lists=False)
            for j in range(len(patch_tile_yx_topleft[i]))] for i in range(self.yx_locations_topleft.shape[0])]            
            
        
        ### retrive all tiles
        all_keys = [self._key_maker(*num) for num in set([(r[0], r[1]) for r in np.concatenate(self.patch_tile_nums, axis=0) ])]
        self.tiles = {k: self.get_tiles_fn(*k) for k in all_keys}
            
        self.incache_tiles_order = {i: k for i, k in enumerate(self.tiles.keys())}
        self.tiles_index = {k: i for i, k in self.incache_tiles_order.items()}
        
        if cuda_device_index is not None:
            self.cuda_device = cp.cuda.Device(cuda_device_index)
            self.use_cuda = True
            self.get_single_patch = self._get_single_patch_cuda
            self.get_multi_patch = self._get_multi_patch_cuda
            with self.cuda_device:
                self.tiles_mat = cp.asarray([self.tiles[self.incache_tiles_order[i]] for i in range(len(self.incache_tiles_order))])
        else:
            self.use_cuda = False
            self.get_single_patch = self._get_single_patch_cpu
            self.get_multi_patch = self._get_multi_patch_cpu
            self.tiles_mat = np.asarray([self.tiles[self.incache_tiles_order[i]] for i in range(len(self.incache_tiles_order))])

        print('Tiling init done for: ', key_image_id)
        
    def _get_single_patch_cpu(self, patch_index):
        # tile key = (image_id, (num_y, num_x), level)
        tiles = self.tiles_mat[[self.tiles_index[self._key_maker(*num)] for num in self.patch_tile_nums[patch_index]]]
        cuts = self.patch_and_tile_cuts[patch_index]        
        # overlap...() returns top_left_1, bottom_right_1, top_left_2, bottom_right_2
        return np.sum(np.array([np.pad(tiles[i][cuts[i][2][0]: cuts[i][3][0], cuts[i][2][1]: cuts[i][3][1]], 
                                      ((cuts[i][0][0], self.yx_patch_size[0] - cuts[i][1][0]), (cuts[i][0][1], self.yx_patch_size[1] - cuts[i][1][1]), (0, 0))) 
                       for i in range(tiles.shape[0])]), axis=0)
    
    def _get_multi_patch_cpu(self, patch_indeces):
        return np.stack([self._get_single_patch_cpu(i) for i in patch_indeces], axis=0)

    def _get_single_patch_cuda(self, patch_index):
        tiles = self.tiles_mat[[self.tiles_index[self._key_maker(*num)] for num in self.patch_tile_nums[patch_index]]]
        cuts = self.patch_and_tile_cuts[patch_index]        
        with self.cuda_device:
            return cp.sum(cp.array([cp.pad(tiles[i][cuts[i][2][0]: cuts[i][3][0], cuts[i][2][1]: cuts[i][3][1]], 
                                          ((cuts[i][0][0], self.yx_patch_size[0] - cuts[i][1][0]), (cuts[i][0][1], self.yx_patch_size[1] - cuts[i][1][1]), (0, 0))) 
                           for i in range(tiles.shape[0])]), axis=0)
    
    def _get_multi_patch_cuda(self, patch_indeces):
        with self.cuda_device:
            return cp.stack([self._get_single_patch_cuda(i) for i in patch_indeces], axis=0)
        
    def get_all_minibatch(self, minibatch_size):
        n = len(self.yx_locations_topleft)
        for i in range(0, n, minibatch_size):
            yield self.get_multi_patch(list(range(i, min(n, i + minibatch_size))))