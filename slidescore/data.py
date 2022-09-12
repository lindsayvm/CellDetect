#
# import matplotlib.pyplot as plt
from PIL import UnidentifiedImageError
import builtins
builtins.mylog = []
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import json
from abc import ABC
import math
import itertools
import copy
import warnings
# warnings.filterwarnings('ignore')
import sys
sys.path.append('../../')
from image.utils import get_matrix_tiling_positions, overlap_2d_arrays_with_point, overlap_2d_arrays_with_global_positions
from image.tiling import Tiling
from slidescore.client import SlidescoreClient
from nn.utils import create_minibatch
import pickle
from skimage.measure import shannon_entropy


class SlidescoreDataset(ABC):
    # defaults
    sample_size = 128
    target_mpp = 0.4545

    # Despite Jan's explanation: The formula log2(max(width, height)) (without the + 1) seems to be correct!! 
    
    # Jan's explanation:    
    # The max level is always log2(max(width, height)) + 1 so if a slide is 131072 (2^17) pixels wide the highest level available is 18.
    # For this slide:
    # Log2(67584) = 16.0443941194
    # Ceiling(16.0443941194) = 17
    # 17 + 1 = 18
    # So max allowed level is 18
    # 
    # Additionally, the max level has to be dependent on the size of the slide.
    # Slide that’s 1x1 pixels only needs 1 level of the pyramid. As it gets bigger you need more levels to capture the detail/be able to zoom in.
    # 
    # And perhaps most importantly: there are 2 sets of levels. 
    # What we’re discussing here is the “DeepZoom” set of levels – each level is twice as big as the previous one and all of them are there. 
    # The slide can contain completely different set of levels (only the highest detail level is the same). 
    # You can see those in levelWidths and levelHeights attributes of the metadata. 
    # They are usually not twice bigger sometimes 4x or other ratios. 
    
    # depricated:
    # each level is twice the width and height of previous level, level 1 is 1x1 pixel. level 17 is (for some slides) the max
    # but better set it to 16. available for most and is better visible.
    # max_zoom_level = 20
    # min_zoom_level = 15  # keep at most 16, not all slides have zoom 17

    def __init__(self, key_path, sample_size=None, target_mpp=None, server=None):
        apitoken = open(key_path, 'r').read().strip()
        
        if server is not None: 
            self.client = SlidescoreClient(apitoken, server=server)
        else:
            self.client = SlidescoreClient(apitoken)
    
        self.scores = pd.DataFrame()
        self.labels = pd.DataFrame()
        self.sample_size = sample_size or SlidescoreDataset.sample_size
        self.target_mpp = target_mpp or SlidescoreDataset.target_mpp


  
    def _collect_metadata(self):
        self.metadata = {image_id: self.client.get_image_metadata(image_id) for image_id in self.scores['ImageID'].unique()}
        return(self.metadata)

    def _calculate_conversions(self):
        for k, v in self.metadata.items():
            v['max_osd_zoom_level'] = math.ceil(math.log2(max(v['level0Width'], v['level0Height']))) ### + 1, see above
            v['mpp'] = (v['mppX'] + v['mppY']) / 2
            v['divide'] = round(self.target_mpp / v['mpp'])
            v['zoom_out'] = round(math.log2(v['divide']))
            v['target_level'] = v['max_osd_zoom_level'] - v['zoom_out']

    # def _calculate_conversions(self):
    #     print(self.scores)
    #     for image_id in self.scores['ImageID'].unique():
    #         v = self.metadata[image_id]
    #         v['max_osd_zoom_level'] = math.ceil(math.log2(max(v['level0Width'], v['level0Height'])))
    #         v['mpp'] = (v['mppX'] + v['mppY']) / 2
    #         v['divide'] = round(self.target_mpp / v['mpp'])
    #         v['zoom_out'] = round(math.log2(v['divide']))
    #         v['target_level'] = v['max_osd_zoom_level'] - v['zoom_out']
    #         return(v)

    def __len__(self):
        return self.labels.shape[0]
    
    def change_target_level(self, zoomout_levels):
        # accounts for when a shallow copy wants to change target mpp
        self.metadata = copy.deepcopy(self.metadata)
        for v in self.metadata.values():
            new_target_level = v['target_level'] - min(v['target_level'], zoomout_levels)
            v['divide'] = v['divide'] * 2 ** (v['target_level'] - new_target_level)
            v['target_level'] = new_target_level
    
    def get_label(self, i):
        return self.labels.loc[i]
    
    def split(self, r=.75, shuffle=True):
        perm = np.random.permutation(len(self))
        len_split1 = int(r * len(self))
        
        split1 = copy.copy(self) # should we do deepcopy ?
        split1.labels = self.labels.iloc[perm[:len_split1], :].reset_index(drop=True)
        
        split2 = copy.copy(self)
        split2.labels = self.labels.iloc[perm[len_split1:], :].reset_index(drop=True)
        return split1, split2
        
    def get_patch(self, image_id, top_left_patch, adjust_point_to_target_zoom=True):
        
        tile_size = self.metadata[image_id]['osdTileSize']
        level = self.metadata[image_id]['target_level']
        patch = np.zeros((self.sample_size, self.sample_size, 3))

        if adjust_point_to_target_zoom:
            div = self.metadata[image_id]['divide']
            top_left_patch = (int(top_left_patch[0] / div), int(top_left_patch[1] / div))
        
        bottom_right_patch = (top_left_patch[0] + self.sample_size, top_left_patch[1] + self.sample_size)

        top_left_tile_nr = [int(top_left_patch[0] / tile_size), int(top_left_patch[1] / tile_size)]
        bottom_right_tile_nr = [int(bottom_right_patch[0] / tile_size), int(bottom_right_patch[1] / tile_size)]
        
        for t in itertools.product(range(top_left_tile_nr[0], bottom_right_tile_nr[0] + 1), range(top_left_tile_nr[1], bottom_right_tile_nr[1] + 1)):
            # now calculate top left and bottom right of the tile in global refrence point
            
            # API: /i/<imageID>/<urlPart>/i_files/<level>/<column>_<row>.jpeg
            try:
                tile = self.client.get_tile(image_id, (t[1], t[0]), level)
            except UnidentifiedImageError:
                print('UnidentifiedImageError', image_id, (t[1], t[0]), level)
                continue
            top_left_tile = (t[0] * tile_size, t[1] * tile_size)
            bottom_right_tile = (top_left_tile[0] + tile.shape[0], top_left_tile[1] + tile.shape[1])
            
            tl_patch, br_patch, tl_tile, br_tile = overlap_2d_arrays_with_global_positions(top_left_patch, bottom_right_patch, 
                                                                                           top_left_tile, bottom_right_tile, 
                                                                                           arguments_are_lists=True)

            patch[tl_patch[0]: br_patch[0], tl_patch[1]: br_patch[1], :] = tile[tl_tile[0]: br_tile[0], tl_tile[1]: br_tile[1], :]
        return patch
        #return np.zeros((2,2,2))
        
        
    def get_patch_enmasse_unsafe(self, image_id, top_left_patch_list, adjust_point_to_target_zoom=True):
        pass

    def get_patch_with_center(self, image_id, center_point, adjust_point_to_target_zoom=True): # (x, y) is the center of the patch
        margin = int(self.sample_size / 2)

        if adjust_point_to_target_zoom:
            div = self.metadata[image_id]['divide']
            center_point = (int(center_point[0] / div), int(center_point[1] / div))

        return self.get_patch(image_id, (center_point[0] - margin, center_point[1] - margin), adjust_point_to_target_zoom=False) # already adjusted


    def get_samples(self, return_locations=False):
        raise NotImplementedError()
    
    def score(self, scoring_fn, return_labels=False):
        scores = []
        
        try: 
            if return_labels:
                for x, y, loc in self.get_samples(return_locations=True):
                    scores.append((scoring_fn(x), y, loc))
            else:
                for x, y, loc in self.get_samples(return_locations=True):
                    scores.append((scoring_fn(x), loc))
        except ValueError:
            warnings.warn('Got an error while processing a sample.')
        
        return scores
    
    def score_minibatch(self, mb_scoring_fn, mb_size, return_labels=False):
        scores = []
        data_iter = create_minibatch(self.get_samples(return_locations=True), mb_size, nested_tuples=1)
        
        c = 0
        try: 
            if return_labels:
                for x, y, loc in data_iter:
                    scores.append((mb_scoring_fn(np.stack(x, axis=0)), y, loc))
                    c += 1
            else:
                for x, y, loc in data_iter:
                    scores.append((mb_scoring_fn(np.stack(x, axis=0)), loc))
                    c += 1
                    #if c > 100:
                    #    break

        except ValueError:
            warnings.warn('Got an error while processing a sample.')
        
        if return_labels:
            return [(s[0][i], s[1][i], s[2][i]) for s in scores for i in range(len(s[0]))]
        else:
            return [(s[0][i], s[1][i]) for s in scores for i in range(len(s[0]))]


class CellDetectionDataset(SlidescoreDataset):    
    gausian_cell_annotation_mask_variance = 12 # 32 # pixels
    tophat_cell_annotation_mask_threshold = 0.3
    empty_patch_shannon_entropy_threshold = 3.5
    sample_ignore_margin = 8


    def __init__(self, key_fpath, annotation_fpath=None, annotation_by=None, 
                 sample_size=None, stride=None, 
                 boxes_question=None, points_question=None, image_id=None, 
                 channel_first=False, y_has_channel=False, entropy_filter=False,
                 annotation_mask='gaussian', server=None):
        super().__init__(key_fpath, sample_size=sample_size, server=server)

       
        if stride is None:
            stride = sample_size - CellDetectionDataset.sample_ignore_margin * 2
        self.stride = stride
        self.channel_first = channel_first
        self.y_has_channel = y_has_channel
        self.entropy_filter = entropy_filter
        
        if annotation_fpath is None:
            self.scores = None
            self.boxes = None
            self.points = None
            return

        # read annotations and metadata
        a = pd.read_csv(annotation_fpath, sep='\t', dtype=str)
        self.scores =  a[a['By'] == annotation_by].copy().reset_index(drop=True)
      
        if image_id:
            self.scores = self.scores[self.scores['ImageID'] == image_id]

        self._collect_metadata()
        self._calculate_conversions() # ??? add k (image id?)
 
        middle = round(sample_size / 2)
        gmesh = np.meshgrid(np.linspace(0, sample_size, sample_size), np.linspace(0, sample_size, sample_size))
        
        variance = (SlidescoreDataset.target_mpp / self.target_mpp) * CellDetectionDataset.gausian_cell_annotation_mask_variance
        gdensity = multivariate_normal(mean=[middle, middle], cov=[[variance, 0], [0, variance]])
        self.gaussian_mask = gdensity.pdf(np.dstack(gmesh))
        self.gaussian_mask = (self.gaussian_mask - self.gaussian_mask.min()) / self.gaussian_mask.max()
        
        t = self.gaussian_mask.max() * CellDetectionDataset.tophat_cell_annotation_mask_threshold
        self.tophat_mask = self.gaussian_mask.copy()
        self.tophat_mask[self.tophat_mask > t] = t
        self.tophat_mask = (self.tophat_mask - self.tophat_mask.min()) / self.tophat_mask.max()
        
        if annotation_mask == 'gaussian':
            self.mask = self.gaussian_mask
        elif annotation_mask == 'tophat':
            self.mask = self.tophat_mask
        else:
            raise ValueError(f'{str(annotation_mask)} not implemented')

        self.boxes = pd.concat([pd.DataFrame([(p['ImageID'], # all coordinates are adjusted to the target level
                                               int(i['corner']['x'] / self.metadata[p['ImageID']]['divide']), 
                                               int(i['corner']['y'] / self.metadata[p['ImageID']]['divide']),
                                               int(i['size']['x'] / self.metadata[p['ImageID']]['divide']),
                                               int(i['size']['y'] / self.metadata[p['ImageID']]['divide']))
                                              for i in json.loads(p['Answer'])], 
                                             columns=['image_id', 'corner_x', 'corner_y', 'size_x', 'size_y']) 
                                for _, p in self.scores[self.scores['Question'] == boxes_question].iterrows()]
                              ).sort_values(by='image_id').reset_index(drop=True)
        self.labels = self.boxes

        try:
            self.points = pd.concat([pd.DataFrame([(p['ImageID'], # all coordinates are adjusted to the target level
                                                    int(i['x'] / self.metadata[p['ImageID']]['divide']),
                                                    int(i['y'] / self.metadata[p['ImageID']]['divide']))
                                                   for i in json.loads(p['Answer'])],
                                                  columns=['image_id', 'x', 'y']) 
                                     for _, p in self.scores[self.scores['Question'] == points_question].iterrows()]
                                   ).sort_values(by='image_id').reset_index(drop=True)
        except ValueError:
            self.points = pd.DataFrame([], columns=['image_id', 'x', 'y'])
            warnings.warn('Warning: found no points to process.')

    def get_samples_with_annotations(self, annotations, return_locations=False):
        for _, b in self.boxes.iterrows():
            ipoints = self.points[self.points.image_id == b.image_id]#.copy().reset_index(drop=True)

            for t in get_matrix_tiling_positions([b.size_y, b.size_x], [self.sample_size, self.sample_size], [self.stride, self.stride]):

                top_left = (b.corner_y + t[0][0], b.corner_x + t[0][1])
                x = self.get_patch(b.image_id, top_left, adjust_point_to_target_zoom=False)

                if self.entropy_filter and shannon_entropy(x.sum(2)) < CellDetectionDataset.empty_patch_shannon_entropy_threshold:
                    continue

                if self.channel_first:
                    x = x.transpose(2, 0, 1)

                if annotations == 'positions' or annotations == 'mask':
                    tpoints_ind = (ipoints.y >= (b.corner_y + t[0][0])) & (ipoints.y < (b.corner_y + t[1][0])) \
                                  & (ipoints.x >= (b.corner_x + t[0][1])) & (ipoints.x < (b.corner_x + t[1][1]))
                    if tpoints_ind.any():
                        # make x and y relative to the tile
                        ys = ipoints[tpoints_ind].y.values - b.corner_y - t[0][0]
                        xs = ipoints[tpoints_ind].x.values - b.corner_x - t[0][1]
                        y = np.stack((ys, xs), axis=0).T
                    else:
                        y = np.array([])

                    if annotations == 'mask':
                        y_mask = np.zeros((self.sample_size, self.sample_size))
                        sample_shapes = ((self.sample_size, self.sample_size),) * len(y)

                        tl1, br1, tl2, br2 = overlap_2d_arrays_with_point(sample_shapes, sample_shapes, y, 
                                                               ((int(self.sample_size / 2), int(self.sample_size / 2)),) * len(y), 
                                                               arguments_are_lists=True)

                        for i in range(tl1.shape[0]):
                            y_mask[tl1[i][0]: br1[i][0], tl1[i][1]: br1[i][1]] = \
                                np.dstack((self.mask[tl2[i][0]: br2[i][0], tl2[i][1]: br2[i][1]], y_mask[tl1[i][0]: br1[i][0], tl1[i][1]: br1[i][1]])).max(axis=-1)

                        if self.y_has_channel:
                            y_mask = np.expand_dims(y_mask, axis=2)
                            if self.channel_first:
                                y_mask = y_mask.transpose(2, 0, 1)

                        y = y_mask

                else:
                    raise ValueError(f'Unknown annotation format: {annotations}. Should be None, "positions", or "mask"')

                if return_locations:
                    yield x, y, (b.image_id, top_left)
                else:
                    yield x, y        

    def get_samples(self, annotations=None, return_locations=False):
        if annotations:
            yield from self.get_samples_with_annotations(annotations, return_locations=return_locations)
        else:
            for _, b in self.boxes.iterrows():
                for t in get_matrix_tiling_positions([b.size_y, b.size_x], [self.sample_size, self.sample_size], [self.stride, self.stride]):

                    top_left = (b.corner_y + t[0][0], b.corner_x + t[0][1])
                    x = self.get_patch(b.image_id, top_left, adjust_point_to_target_zoom=False)

                    if self.entropy_filter and shannon_entropy(x.sum(2)) < CellDetectionDataset.empty_patch_shannon_entropy_threshold:
                        continue

                    if self.channel_first:
                        x = x.transpose(2, 0, 1)

                    if return_locations:
                        yield x, None, (b.image_id, top_left)
                    else:
                        yield x, None        

    def get_label(self, i):
        pass


class WSICellDetectionDataset(CellDetectionDataset):
    def __init__(self, key_fpath, study_id, slide_id=None,
                 sample_size=None, stride=None, wsi_margin=None,
                 channel_first=False, entropy_filter=False,
                 annotation_mask='gaussian', server=None):
        
        super().__init__(key_fpath, sample_size=sample_size, stride=stride, 
                         channel_first=channel_first, entropy_filter=entropy_filter, server=server)
                
        self.study_id = study_id
        
        if wsi_margin is None:
            wsi_margin = self.sample_size
        
        self.wsi_margin = wsi_margin
        
        images = self.client.get_images(study_id)
        
        if slide_id is None:
            self.metadata = {img['id']: self.client.get_image_metadata(img['id']) for img in images}
        else:
            self.metadata = {slide_id: self.client.get_image_metadata(slide_id)}
        
        self._calculate_conversions()

        # all coordinates are adjusted to the target level
        self.boxes = pd.DataFrame(
                [(k, self.wsi_margin, self.wsi_margin, 
                  int(v['level0Width'] / v['divide']) - self.wsi_margin, int(v['level0Height'] / v['divide']) - self.wsi_margin)
                 for k, v in self.metadata.items()], 
                columns=['image_id', 'corner_x', 'corner_y', 'size_x', 'size_y']
            ).sort_values(by='image_id').reset_index(drop=True)
        
        self.labels = self.boxes

    
class CellClassificationDataset(SlidescoreDataset):   
    def __init__(self, key_fpath, annotation_fpath=None, annotation_by=None, 
                 sample_size=None, channel_first=False, shuffle=False, filter_classes=None, server=None):
        super().__init__(key_fpath, sample_size=sample_size, server=server)
        self.channel_first = channel_first
        self.shuffle = shuffle
    
    
        if annotation_fpath is None:
            self.scores = None
            self.labels = None
            self.annotation_classes = None
            return 
        
        if isinstance(annotation_by, str):
            annotation_by = [annotation_by]
        
        # read annotations and metadata
        a = pd.read_csv(annotation_fpath, sep='\t', dtype=str)
        self.scores = a[a['By'].apply(lambda b: b in annotation_by)].copy().reset_index(drop=True)
        self.annotation_classes = {v: i for i, v in enumerate(self.scores['Question'].unique())}
        self.labels = pd.concat(
            [pd.DataFrame(
                [(p['ImageID'], p['Question'], i['x'], i['y']) for i in json.loads(p['Answer']) if i.get('x', False)], 
                columns=['image_id', 'label', 'x', 'y'])
             for _, p in self.scores.iterrows()]).sort_values(by='image_id').reset_index(drop=True)
        
        if filter_classes is not None:
            self.labels = self.labels[self.labels['label'].apply(lambda s: s in filter_classes)].reset_index(drop=True)
        
        self._collect_metadata()
        self._calculate_conversions()

    def get_sample(self, i):
        s = self.get_label(i)
        x = self.get_patch_with_center(s.image_id, (s.y, s.x), adjust_point_to_target_zoom=True)
        
        if self.channel_first:
            x = x.transpose(2, 0, 1)
        
        if self.annotation_classes is not None:
            return x, self.annotation_classes[self.get_label(i).label]
        else:
            return x, None
    
    def get_samples(self, return_locations=False):
        if self.shuffle:
            itr = np.random.permutation(len(self))
        else:
            itr = range(len(self))
        
        if return_locations:
            for i in itr:
                yield self.get_sample(i), self.get_label(i)
        else:
            for i in itr:
                yield self.get_sample(i)


class WSICellClassificationDatasetFromCache(CellClassificationDataset):   
    def __init__(self, key_fpath, image_id, sample_size, cached_tiles, cell_centers,
                 channel_first=False, minibatch_size=None, server=None, cuda_device_index=None):
        
        super().__init__(key_fpath=key_fpath, annotation_fpath=None, annotation_by=None, 
                 sample_size=sample_size, shuffle=False, server=server)
        
        self.cuda_device_index = cuda_device_index
       
        
        if minibatch_size is None:
            minibatch_size = 1
        self.minibatch_size= minibatch_size
        self.channel_first = channel_first
        
        self.metadata = {image_id: self.client.get_image_metadata(image_id)}
        self._calculate_conversions()
        tile_size = self.metadata[image_id]['osdTileSize']
        level = self.metadata[image_id]['target_level']
        self.cell_centers = (cell_centers / self.metadata[image_id]['divide']).astype(int)
        
        
        self.client.load_cache(tiles=cached_tiles)
        self.tiling = Tiling(yx_patch_size=(sample_size, sample_size), yx_tile_size=(tile_size, tile_size), 
                 get_tiles_fn=self.client.get_tile, key_image_id=image_id, key_level_value=level,
                 yx_locations_centers=self.cell_centers, 
                 cuda_device_index=cuda_device_index)

    def get_samples(self):
        for x in self.tiling.get_all_minibatch(self.minibatch_size):
            if self.channel_first:
                x = x.transpose(0, 3, 1, 2)
            yield x, [None]


class DoubleCellClassificationDataset:
    def __init__(self, c0, c1, shuffle=False, labels_from=0):
        if len(c0) != len(c1):
            raise ValueError('The two datasets must have equal lengths.')
        else:
            self._n = len(c0)
        
        self.c0 = c0
        self.c1 = c1        
        self.labels_from = labels_from
        self.shuffle = shuffle

    def get_sample(self, i):
        x0, y0 = self.c0.get_sample(i)
        x1, y1 = self.c1.get_sample(i)
        return (x0, x1), y0 if self.labels_from == 0 else y1
    
    def get_samples(self, return_locations=False):
        if self.shuffle:
            itr = np.random.permutation(self._n)
        else:
            itr = range(self._n)
        
        if return_locations:
            for i in itr:
                yield self.get_sample(i), self.c0.get_label(i) if self.labels_from == 0 else self.c1.get_label(i)
        else:
            for i in itr:
                yield self.get_sample(i)

    def __len__(self):
        return self._n
    
    def split(self):
        raise NotImplementedError()
