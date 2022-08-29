import warnings
warnings.filterwarnings("ignore")

#
import asyncio
import concurrent.futures as futures
import copy
import gc
import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
from skimage import morphology, filters
import torch
from multiprocessing import Process
import torch.multiprocessing
from collections import namedtuple
#
import config as gconfig
from tracking import TaskTracker
sys.path.append(gconfig.base_path)
from slidescore.data import CellDetectionDataset, WSICellDetectionDataset, CellClassificationDataset, WSICellClassificationDatasetFromCache
from slidescore.client import SlidescoreClient
import nn.models as models
from image.data import HistoDataset
import segmentation.segmentation_models_pytorch as smp

#
torch.multiprocessing.set_sharing_strategy('file_system')
torch.multiprocessing.set_start_method('spawn', force=True)
torch.cuda.set_per_process_memory_fraction(1 / (gconfig.max_running_tasks + 0.25))

#
loop = asyncio.get_event_loop()
cd_task, cc_task = None, None 
tracker = None 
c = type('c', (object, ), {})


def get_picklable_config():
    c1 = c()
    for s in dir(gconfig):
        if '__' not in s:
            setattr(c1, s, getattr(gconfig, s))
    return c1
    # return type('c', (c, ), {s: getattr(gconfig, s) for s in dir(gconfig) if '__' not in s})()
    # attrs = [s for s in dir(gconfig) if '__' not in s]
    # return namedtuple('c', attrs)(*[getattr(gconfig, s) for s in attrs])


def cell_detection(image_id, config, load_from_previous=True):
    
    results_save_path = config.cd_results_path.format(image_id)
    cache_save_path = config.tiles_cache_save_path.format(image_id)
    device = config.default_device
    
    if load_from_previous:
        try:
            with open(results_save_path, 'rb') as f:
                print(results_save_path + ' found previous cd results...')
                return pickle.load(f)
        except FileNotFoundError:
            print(results_save_path + ' not found, running cd...')

    detection_dataset = CellDetectionDataset(
        key_fpath=config.api_token_path,
        annotation_fpath=config.cd_targets_path, annotation_by=config.cd_targets_annotation_by,
        boxes_question=config.cd_target_box_question, points_question='',
        image_id=image_id,
        sample_size=config.cd_sample_size, channel_first=True, entropy_filter=True,
        server=config.server)
    
    try:
        detection_dataset.client.load_cache(path=cache_save_path)
    except:
        pass
            
    def output2points(x_batch):
        # Value 2. for sigma and 0.3 for threshold were chosen by manually inspecting results
        f_batch = [filters.gaussian(np.squeeze(x), sigma=config.cd_gaussiona_filter_sigma) for x in x_batch]
        p_batch = [np.stack(morphology.local_maxima((f > config.cd_probability_threshold) * f, indices=True)).T for f in f_batch]
        return [p[np.all(p > CellDetectionDataset.sample_ignore_margin , axis=1)] for p in p_batch]

    def sfn(_x):
        x = torch.from_numpy(_x).float().to(device)
        return output2points(model(x).detach().to("cpu").numpy())

    with torch.no_grad():
        model = models.load_state(models.resnet152unet(config.cd_sample_size, pretrained=False), config.cd_model_path, device=config.default_device)
        #model = models.load_state(smp.Unet('densenet161', encoder_weights=None), config.cd_model_path, device=config.default_device)

    model.to(device)
    with torch.no_grad():
        scores = detection_dataset.score_minibatch(sfn, config.cd_mb_size)

    adjusted_scores = {s[1][0]: [] for s in scores}
    _ = [adjusted_scores[s[1][0]].append((np.tile(s[1][1], (s[0].shape[0], 1)) + s[0]) * detection_dataset.metadata[s[1][0]]['divide']) for s in scores]
    adjusted_scores = {k: np.concatenate(v, axis=0) for k, v in adjusted_scores.items()}

    with open(results_save_path, 'wb') as f:
        pickle.dump(adjusted_scores, f)
    
    detection_dataset.client.save_cache(path=cache_save_path)
    return adjusted_scores


async def spawn_cd(image_id):
    with futures.ProcessPoolExecutor(max_workers=1) as pool:
        await loop.run_in_executor(pool, cell_detection, image_id, get_picklable_config())

    print('- cd done for: ', image_id)
    await tracker.finish_cd(image_id)


def cell_classification(image_id, config, load_from_previous=True):
    
    cd_results_path = config.cd_results_path.format(image_id)
    results_save_path = config.cc_results_path.format(image_id)
    cache_load_path = config.tiles_cache_save_path.format(image_id)
    device = config.default_device
    if load_from_previous:
        try:
            with open(results_save_path, 'rb') as f:
                print(results_save_path + ' found previous cc results...')
                return pickle.load(f)
        except FileNotFoundError:
            print(results_save_path + ' not found, running cc...')

    #cd_results_path = config.cd_results_path.format(image_id)
    cd_data = WSICellDetectionDataset(
        key_fpath=config.api_token_path, study_id=config.study_id, slide_id=image_id,
        sample_size=config.cd_sample_size, channel_first=True, entropy_filter=True,
        server=config.server)

    try:
        cd_data.client.load_cache(path=cache_load_path)
    except:
        print('!!!! could not find tiles cache: ', cache_load_path)
    
    with open(cd_results_path, 'rb') as f:
        cd_scores = pickle.load(f)

    classification_dataset = WSICellClassificationDatasetFromCache(
        key_fpath=config.api_token_path, image_id=image_id, sample_size=config.cc_sample_size, 
        cached_tiles=cd_data.client._tiles, cell_centers=cd_scores[image_id],
        channel_first=True, minibatch_size=config.cc_mb_size, server=config.server, cuda_device_index=0) ### GET PROPER DEVICE!!!!        
    
    classification_dataset.labels = pd.DataFrame(
        np.concatenate([np.concatenate((np.tile([k, -1], (v.shape[0], 1)), v), axis=1) for k, v in cd_scores.items()]),
        columns=['image_id', 'label', 'y', 'x'])
    classification_dataset.scores = pd.DataFrame(list(cd_scores.keys()), columns=['ImageID'])
    
    cc_data = HistoDataset(classification_dataset.get_samples, provides_minibatches=True, im_normalize=True, output_type='long', cupy_scoring_mode=True)

    with torch.no_grad():
        #cc_model = models.load_state(models.iv3(config.cc_num_classes, pretrained=False), config.cc_model_path)
        model = models.load_state(models.densenet161(config.cc_num_classes, pretrained=False), config.cc_model_path, device=device)

    preds = []
    model.to(device)
    model.eval()
    c = 0
    for _x, _ in cc_data:
        x = _x.to(device)
        with torch.no_grad():
            preds.append(model(x).detach().to("cpu").numpy())
        c += 1
        #if c >= 10:
        #    break

    results = {'preds': np.concatenate(preds), 'cell_labels_df': classification_dataset.labels}

    with open(results_save_path, 'wb') as f:
        pickle.dump(results, f)
    
    return results


async def spawn_cc(image_id):
    with futures.ProcessPoolExecutor(max_workers=1) as pool:
        await loop.run_in_executor(pool, cell_classification, image_id, get_picklable_config())
    
    print('- cc done for: ', image_id)
    await tracker.finish_cc(image_id)


def get_updated_boxes():
    a = pd.read_csv(gconfig.cd_targets_path, sep='\t')
    images = a[a['By'] == gconfig.cd_targets_annotation_by]['ImageID'].astype(str).tolist()
    return images


async def _run():
    
    if not os.path.isdir(gconfig.results_path):
        print('making dir: ' + gconfig.results_path)
        os.makedirs(gconfig.results_path)

    while True:
        tracker.update_all_ids(get_updated_boxes())
        image_id, task_type = await tracker.get_next_slide()
        
        print('starting with: ', image_id, task_type)
        if task_type == 'cd':            
            await tracker.start_cd(image_id)
            cd_task = loop.create_task(spawn_cd(image_id))
        elif task_type == 'cc':            
            await tracker.start_cc(image_id)
            cc_task = loop.create_task(spawn_cc(image_id))
        elif task_type == 'finished_cd':
            # This situation occurs only when all cd tasks are asigned 
            # but are not finished yet, and all assigned cc tasks are finished.
            print('was waiting for cd to finish.')
        elif task_type == 'finished_cc':
            print('task assignment finished.')
            break
        else:
            raise ValueError()
        
        gc.collect()
        torch.cuda.empty_cache()
    
    await tracker.finished()
    print('all finished.')


# cell detecttion does not, use im_normalize but cell classification does... (!)
# See https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
if __name__ == '__main__':
    #asyncio.run()
    run_path = sys.argv[1] + '/' + sys.argv[2] + '/'  # [:sys.argv[1].rfind('/') + 1]
    gconfig.results_path = run_path
    gconfig.cd_targets_path = run_path + 'mrr.txt'
    gconfig.progress_file_path = run_path + 'progress.json'
    gconfig.cd_results_path = run_path + 'cell_detection_results_img{}.pkl'
    gconfig.cc_results_path = run_path + 'cell_classification_results_img{}.pkl'
    gconfig.tiles_cache_save_path = run_path + 'odcc_scoring_cache_img{}.pkl'
    #
    tracker = TaskTracker(get_updated_boxes(), gconfig.progress_file_path)
    loop.run_until_complete(_run())
