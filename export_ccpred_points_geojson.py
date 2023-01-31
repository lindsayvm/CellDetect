import os
import pandas as pd
from slidescore.client import SlidescoreClient
from slidescore.data import CellDetectionDataset 
import config
import pickle
from scipy.special import softmax
import json

parent_dir = '/data/lindsay/CellDetect/' 
os.chdir(parent_dir)

annotation_fpath= 'IID/mrr_scoring_mask.txt' #has to have rectangle annotations
a = pd.read_csv(annotation_fpath, sep='\t', dtype=str)
###!!!! select for annotated by lindsay so you dont have empty rectangles
a = a[a.By == 'l.leek@nki.nl']


patientNr = "386687" #"386686"#"386685" #"386683"#"386672"   #386688, 
directory = patientNr.strip() + "_annotated_regions"
t_Nr = a[a.ImageID ==  patientNr]['Image Name'].iloc[0].replace(" ","_")
print(t_Nr)

# pd.options.display.max_rows
# print(a[(a["ImageID"] == patientNr) & (a["Question"]=='Area with point annotations')]["Answer"])


#read api key and use it to connect to slidescore and download metadata for all images
with open('IID/IID_slidescore_training.key', 'r') as f:
    apitoken = f.read().strip()

client = SlidescoreClient(apitoken, server='https://slidescore.nki.nl')
metadata = {i: client.get_image_metadata(i) for i in a['ImageID'].unique()} #Change to ImageID

with open('IID/cell_classification_results_img'+patientNr+'.pkl', 'rb') as f:
    d = pickle.load(f)
    preds = d['preds']
    scores = d['cell_labels_df']
    
    
#uncertain_threshold = .49
uncertain_threshold = .60
uncertain = softmax(preds, axis=1).max(axis=1) < uncertain_threshold

scores.x, scores.y = scores.x.astype(int), scores.y.astype(int) 

objects = CellDetectionDataset(key_fpath='IID/IID_slidescore_training.key',
                annotation_fpath='IID/mrr_scoring_mask.txt',                
                annotation_by='l.leek@nki.nl', 
                sample_size=config.cc_sample_size, #add sample size to determine stride,
                image_id=patientNr, #add image_id
                boxes_question='Area with point annotations', points_question='')


#get preds from annotated regions
box = objects.boxes.loc[0]
d = objects.metadata[patientNr]['divide']
corners = {'tly': box.corner_y * d, 'tlx': box.corner_x * d, 'bry': (box.corner_y + box.size_y) * d, 'brx':(box.corner_x + box.size_x) * d}
preds_box = scores[(scores.label == 'model_tumor') & (scores.image_id == patientNr) &
           (scores.x >= corners['tlx']) & (scores.x <= corners['brx']) & 
           (scores.y >= corners['tly']) & (scores.y <= corners['bry'])].copy()

preds_box = preds_box[["x","y"]]
preds_box_np = preds_box.to_numpy()
preds_box_ls = preds_box_np.tolist()

t_Nr+'_HE_2class_tumor_img'+patientNr+".json"

geojson = {'type': 'FeatureCollection', 'features': []}
feature = {'type': 'Feature', 
           'geometry': {'type':'MultiPoint',
                        'coordinates':[]}}

# fill in the coordinates 
feature['geometry']['coordinates'] = preds_box_ls
# add this feature (convert dataframe row) to the list of features inside our dict
geojson['features'].append(feature)


with open(t_Nr+'_2class_tumor_img'+patientNr+".json", 'w', encoding='utf-8') as f:
    json.dump(geojson, f , sort_keys=True)
    f.write("\n")