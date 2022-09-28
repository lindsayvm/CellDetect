import numpy as np
import pickle
with open("/home/l.leek/src/CellDetect/IID/cell_detection_results_img386667.pkl", "rb") as f:
    imgList = pickle.load(f)

preds = imglist['pred']
print(preds)

exit()
print(dict(list(imgList.items())))
print(len(list(imgList.items())[0][1][0])) #512
# print((list(imgList.items())[0][1])) #512
# print("The length of the dictionary is {}".format(len(imgList)))

def count_keys(dict_, counter=0):
    for each_key in dict_:
        if isinstance(dict_[each_key], dict):
            # Recursive call
            counter = count_keys(dict_[each_key], counter + 1)
        else:
            counter += 1
    return counter

print('The length of the nested dictionary is {}'.format(count_keys(imgList)))

import pickle
import numpy as np
import matplotlib.pyplot as plt

images = np.load("/home/l.leek/src/CellDetect/IID/odcc_scoring_cache_img386667.pkl",allow_pickle=True) 
image = images[2]
img = np.reshape(image,(32,32,3))
im = Image.fromarray(img)