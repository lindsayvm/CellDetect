
#Are you in ptolemaeus? Are you in singularity? Are you in conda env CellDetect?

#import os
#os.chdir('/home/l.leek/src/')
#print(os.getcwd())

from random import sample
from scipy.special import softmax
import pickle
import pandas as pd
import numpy as np
import json
import sys
import collections 
from sklearn.neighbors import KernelDensity
#pip install scikit-learn==0.24.1
from skimage import measure
from scipy.stats import entropy
import matplotlib.pyplot as plt
from multiprocessing import Pool
import functools
from matplotlib.colors import ListedColormap
import cv2 
#pip install opencv-python==3.4.11.39 #3.4.11.41, 3.4.11.43, 3.4.11.45
# >>> import cv2
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/home/l.leek/.local/lib/python3.8/site-packages/cv2/__init__.py", line 5, in <module>
#     from .cv2 import *
from collections import defaultdict
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.affinity import translate
import fiona
import geopandas as geopd

from slidescore.client import SlidescoreClient
from slidescore.data import CellDetectionDataset 
import config



print("installation")
#problem with installing?
#singularity exec --nv /home/l.leek/docker_singularity_images/u20c114s.sif      bash -c "pip install geopandas; python3 export_ccpred_geojson.py"
#data can be found in /home/l.leek/src/CellDetect/


#Get large segmentation rectangle annotations
annotation_fpath= '/home/l.leek/data/IID/annotations/mrr.txt' 

a = pd.read_csv(annotation_fpath, sep='\t', dtype=str)
a = a[a.By == 'l.leek@nki.nl'] #!!!! select for annotated by lindsay so you dont have empty rectangles

#Get metadata
with open('/home/l.leek/data/IID/apikey/IID_slidescore_training.key', 'r') as f:
    apitoken = f.read().strip()

client = SlidescoreClient(apitoken, server='https://slidescore.nki.nl')
metadata = {i: client.get_image_metadata(i) for i in a['ImageID'].unique()}
 #Change to ImageID; scores --> a

#get cell classification results full: cell_classification_results_img386688  partial: cell_classification_results_img386667
with open('/home/l.leek/data/IID/cell_classification_results_img386688.pkl', 'rb') as f:
    d = pickle.load(f)
    preds = d['preds']
    scores = d['cell_labels_df']

#Compute softmax for rows (thus summed 1) and take max. If max pred is smaller than 0.6 we consider that uncertain
uncertain_threshold = .60
uncertain = softmax(preds, axis=1).max(axis=1) < uncertain_threshold
#print(uncertain.sum())

#change the labels of scores into the most likely preds
labs = ['model_tumor', 'model_lymphocyte']#, 'model_fibroblast', 'model_epi_endo_thelial', 'model_other']
plabs = np.array([labs[i] for i in np.argmax(preds, axis=1)])
plabs[uncertain] = 'model_uncertain'
scores['label'] = plabs


#Check labels in scores are tumor lymph and uncertain
#print(np.unique(scores["label"])) 

#Cannot multiply chars, so change to int
scores.x, scores.y = scores.x.astype(int), scores.y.astype(int) 

#Select patient
image_id = ''.join(np.unique(scores['image_id']))
print(image_id)

#info on tissue boxes
objects = CellDetectionDataset(key_fpath='/home/l.leek/data/IID/apikey/IID_slidescore_training.key',
                annotation_fpath='/home/l.leek/data/IID/annotations/mrr.txt',                
                annotation_by='l.leek@nki.nl', 
                sample_size=config.cc_sample_size, #add sample size to determine stride,
                image_id=image_id, #add image_id
                boxes_question='target_scoring_rectangle', points_question='')


# make points heatmap

resolution_divide = 50
def points2heatmap(l, b):
    # at resolution 200, max 20 tumor are in one cell, let's say 25 in the brightest.
    # so you can cap at (resolution_divide / 200)^2 * 25
    # remember higher resolution_divide = less zoomed in = more cells
    max_cells = (resolution_divide / 200)**2 * 25
    d = objects.metadata[b.image_id]['divide']
    corners = {'tly': b.corner_y * d, 'tlx': b.corner_x * d, 'bry': (b.corner_y + b.size_y) * d, 'brx':(b.corner_x + b.size_x) * d}
    s = scores[(scores.label == l) & (scores.image_id == b.image_id) &
               (scores.x >= corners['tlx']) & (scores.x <= corners['brx']) & 
               (scores.y >= corners['tly']) & (scores.y <= corners['bry'])].copy()
    h = np.zeros((int((corners['bry'] - corners['tly']) / resolution_divide) + 1, int((corners['brx'] - corners['tlx']) / resolution_divide) + 1))
    def ch(r):
        _y = int((r.y - corners['tly'] - 0.5) / resolution_divide)
        _x = int((r.x - corners['tlx'] - 0.5) / resolution_divide)
        h[_y][_x] += 1
    _ = s.apply(ch, axis=1)
    h[h> max_cells] = max_cells
    h = h * 254 / max_cells
    return h.astype(int)

#Tumor cells
print("points2heatmap")
h = points2heatmap('model_tumor', objects.boxes.loc[0])


ph = h#[500:600, 150:250]
figsize = 1/20
#plt.figure(figsize=(ph.shape[1] * figsize, ph.shape[0] * figsize))
#### plt.contourf(ph[::-1, :])
#plt.imshow(ph > 10)
#plt.savefig('/home/l.leek/data/IID/output_mask/figures/tumorcells_points_map.png')
#plt.show()


def cartesian(x1, x2):
    return np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))])

def _f(args):
    return np.exp(args[0].score_samples(args[1]))

def points2smooth(l, b, bandwidth=0.5, num_process=1):
    d = objects.metadata[b.image_id]['divide']
    corners = {'tly': b.corner_y * d, 'tlx': b.corner_x * d, 'bry': (b.corner_y + b.size_y) * d, 'brx':(b.corner_x + b.size_x) * d}
    s = scores[(scores.label == l) & (scores.image_id == b.image_id) &
               (scores.x >= corners['tlx']) & (scores.x <= corners['brx']) & 
               (scores.y >= corners['tly']) & (scores.y <= corners['bry'])].copy()
    
    s.x = s.x / resolution_divide
    s.y = s.y / resolution_divide
    # for res_div = 500, bandwidth 0.5 was ok 
    d = KernelDensity(kernel='gaussian', bandwidth=bandwidth, algorithm='ball_tree', leaf_size=20).fit(s[['y', 'x']])
    gy = np.arange(corners['tly'] / resolution_divide , corners['bry'] / resolution_divide)
    gx = np.arange(corners['tlx'] / resolution_divide , corners['brx'] / resolution_divide)
    grid = cartesian(gy, gx)

    if num_process < 2:
        m = _f((d, grid)).reshape((gx.shape[0], gy.shape[0])).T
    else:
        pool = Pool(num_process)
        part_size = int(grid.shape[0] / num_process) + 1
        g_parts = [(grid[i * part_size: (i + 1) * part_size, :], d) for i in range(num_process)]
        m = np.concatenate(pool.map(_f, g_parts)).reshape((gx.shape[0], gy.shape[0])).T
    return m / m.max()

print("points2smooth")
m = points2smooth('model_tumor', objects.boxes.loc[0], bandwidth=2, num_process=8) 
# points2smooth
# Traceback (most recent call last):
#   File "export_ccpred_geojson.py", line 168, in <module>
#     m = points2smooth('model_tumor', objects.boxes.loc[0], bandwidth=2, num_process=8) 
#   File "export_ccpred_geojson.py", line 159, in points2smooth
#     m = _f((d, grid)).reshape((gx.shape[0], gy.shape[0])).T
# ValueError: cannot reshape array of size 273141 into shape (1417,1705)


pm = m#[500:600, 150:250]# > 0.00001
figsize = 1/20
plt.figure(figsize=(pm.shape[1] * figsize, pm.shape[0] * figsize))
plt.contourf(pm[::-1, :])
# plt.imshow(pm)
plt.savefig('/home/l.leek/data/IID/output_mask/figures/tumorcells_mask.png')
#plt.show()




#make smooth heatmap plot + contour

# th for resolution divide 200, bandwidth 0.5 was 0.05 
# and for rd 50 bw 1.5 was 0.06
th = 0.06   #the higher the more contours
contours = measure.find_contours(m, th, fully_connected='low', positive_orientation='high')
#print(len(contours))

# Display the image and plot all contours found
fig, ax = plt.subplots(figsize=(pm.shape[1] * figsize, pm.shape[0] * figsize))
ax.imshow(pm, cmap=plt.cm.gray)
# ax.contourf(pm)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='green')

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('/home/l.leek/data/IID/output_mask/figures/tumorcells_originalHE_and_mask.png')
#plt.show()


#adjust contours to resolution divide
c_adjusted = [c * resolution_divide for c in contours]
MultiPolygon([Polygon(c) for c in c_adjusted])


_, m_thresh = cv2.threshold(m, th, 1, cv2.THRESH_BINARY)
m_thresh = cv2.convertScaleAbs(m_thresh)



#Make it 8-bit
# from PIL import Image
# def pixelate(input_file_path, pixel_size):
#     image = Image.open(input_file_path)
#     image = image.resize(
#         (image.size[0] // pixel_size, image.size[1] // pixel_size),
#         Image.NEAREST
#     )
#     image = image.resize(
#         (image.size[0] * pixel_size, image.size[1] * pixel_size),
#         Image.NEAREST
#     )
#     return image

# img8bit=pixelate("/home/l.leek/src/CellDetect/IID/output_mask/figures/tumorcells_mask.png",8)
# img8bit = np.array(img8bit)
#print(img8bit)

# figsize=1/2000
# plt.figure(figsize=(img8bit.shape[1] * figsize, img8bit.shape[0] * figsize))
# plt.contourf(img8bit[::-1, :])
# plt.show()
# plt.savefig('/home/l.leek/src/CellDetect/IID/output_mask/figures/mask_binary8bit.png')

# apply thresholding to convert grayscale to binary image.
#print(np.unique(img8bit)) 
#ret,thresh_img = cv2.threshold(img8bit,127.5,255,cv2.THRESH_BINARY)
#print(np.unique(thresh_img)) 

#make sure it s an numpy array
#thresh_img = np.array(thresh_img)

# #normal image
# img, cons, hiers = cv2.findContours(m, m_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# #8bit image
# img, cons, hiers = cv2.findContours(img8bit, m_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# #binary 8bit image
# img, cons, hiers = cv2.findContours(thresh_img, m_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# #no image
img, cons, hiers = cv2.findContours(m_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# len([c for c in cons if c.shape[0] > 1])
# cons[2].shape
# len(cons), hiers.shape
# hiers
# [c * resolution_divide for c in cons]

def contours2multiploygon(cv2contours, cv2hierarchy, min_area=0.1):
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert cv2hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(cv2hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(cv2contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(cv2contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            if min_area > 0.:
                holes = [c[:, 0, :] for c in cnt_children[idx] if cv2.contourArea(c) >= min_area]
            else:
                holes = [c[:, 0, :] for c in cnt_children[idx]]
            poly = Polygon(shell=cnt[:, 0, :], holes=holes)
            all_polygons.append(poly)
    all_polygons = MultiPolygon(all_polygons)
    return all_polygons

mpoly = contours2multiploygon([c * resolution_divide for c in cons], hiers)
# valid_indexes = [c.shape[0] > 2 for c in cons]
# mpoly = contours2multiploygon([c for c in cons if c.shape[0] > 2], hiers[:, [c.shape[0] > 2 for c in cons], :])
#mpoly

# without shift was [[[[62000.0, 78400.0], [62000.0, 78800.0]
# with shift became [[[[69414.0, 101937.0], [69414.0, 102337.0]
b = objects.boxes.loc[0]
mpoly_box_offest = translate(mpoly, xoff=b.corner_x * objects.metadata[b.image_id]['divide'], yoff=b.corner_y * objects.metadata[b.image_id]['divide'])

exjson = str({k:v for k,v in geopd.GeoSeries([mpoly_box_offest]).__geo_interface__['features'][0].items() if k in ['type', 'properties', 'geometry']})
exjson = exjson.replace(',)', ')').replace('(', '[').replace(')', ']').replace('\'', '\"')


with open('/home/l.leek/data/IID/output_mask/T19_07579_II6_HE_2class_tumor_img386688.json', 'w') as f:
    f.write(exjson)


schema = {
    'geometry': 'MultiPolygon',
    'properties': {'id': 'int'},
}

with fiona.open('/home/l.leek/data/IID/output_mask/T19_07579_II6_HE_2class_tumor_img386688.shp', 'w', 'ESRI Shapefile', schema) as c:
    ## If there are multiple geometries, put the "for" loop here
    c.write({'geometry': mapping(mpoly), 'properties': {'id': 1} })