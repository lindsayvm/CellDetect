##
default_device = 'cuda'
server = 'https://slidescore.nki.nl'

## initialization
cd_model_path = '/home/l.leek/src/CellDetect/IID/cell_detection_weights.pkl'
cc_model_path = '/home/l.leek/src/CellDetect/IID/cell_classification_weights.pkl'
cd_sample_size = 256
cc_sample_size = 256
cd_mb_size = 32
cc_mb_size = 128

## run params
cd_targets_annotation_by = 'l.leek@nki.nl' #Overwritten annotations of joyce and marcos on l.leek
cd_target_box_question = 'Area with point annotations'
api_token_path = '/home/l.leek/src/CellDetect/IID/IID_slidescore_training.key'
study_id = 1844 # insert study id
run_all = 0 # if bigger than zero, run_all number of slides will be run
max_running_tasks = 1

## model params
cd_gaussiona_filter_sigma = 2.0
cd_probability_threshold = 0.5
cc_output_labs = ['model_tumor', 'model_lymphocyte', 'model_fibroblast', 'model_other', 'model_red', 'model_normal']
cc_num_classes = len(cc_output_labs)


