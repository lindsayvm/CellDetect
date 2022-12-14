cd src/CellDetect/ 

###FIRST YOU HAVE TO REQUEST A GPU 

#interactive
singularity exec --nv /home/l.leek/docker_singularity_images/u20c114s.sif /bin/bash
singularity exec --nv --no-home /home/l.leek/docker_singularity_images/u20c114s.sif /bin/bash
#### THIS IS WHERE YOU ALSO INSTALL YOUR PACKAGES
#exit with ctrl+ d

#same: interactive
singularity shell --nv /home/l.leek/docker_singularity_images/u20c114s.sif

#If some packages are missing from singularity container. (The above interaction command directs to an environment that uses a different python location and packages)
singularity exec --nv /home/l.leek/docker_singularity_images/u20c114s.sif \
    bash -c "python3 -m pip install torch torchvision matplotlib spacy attrs"




#Task 1: cell detection. Training of model for cell detection has been done by Siamak

#Task 2: Train classification model weights with annotations from slidescore
singularity exec --nv /home/l.leek/docker_singularity_images/u20c114s.sif \
    bash -c "python3 -m train.py"

#Task 3: Score slides
cd src/CellDetect/ 
singularity exec --nv /home/l.leek/docker_singularity_images/u20c114s.sif  \
    bash -c "python3 -m classification.pipeline.scoring_cupy_single /home/l.leek/src/CellDetect/IID/"
    
#Task 4: make a mask of slides
singularity exec --nv /home/l.leek/docker_singularity_images/u20c114s.sif \
    bash -c "python3 export_ccpred_geojson.py"

#Task 5: make confusion_matrix
singularity exec --nv /home/l.leek/docker_singularity_images/u20c114s.sif \
    bash -c "python3 ##################"





#SIAMAK##############################################################################################################################################

#!/bin/bash
# sbatch --gres=gpu:1 --cpus-per-task=12 --mem=64G --time=D-HH:MM:SS \ 
# sbatch --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=2-00 ./src/rnki/classification/pipeline/run.sh src/rnki/classification/pipeline/runs/yb_run1_4march2022_first30oo90/


#run command in singularity
#singularity exec --nv /path/to/singularity_image.sif \
#    bash -c "cd /path/to/repo-root/classification/pipeline/ && python3  scoring_cupy_single.py $PWD $1"


#singularity exec --nv /home/s.hajizadeh/docker_singularity_images/siamakg_23feb2022.sif \
#    bash -c "cd /home/s.hajizadeh/src/rnki/classification/pipeline/ && python3  scoring_cupy_cc.py $PWD $1"

#singularity exec --nv /home/s.hajizadeh/docker_singularity_images/siamakg_23feb2022.sif \
#    bash -c "cd /home/s.hajizadeh/src/rnki/classification/pipeline/ && python3  scoring_cupy_single.py $PWD $1"

#bash -c "pwd && echo $1"

#/home/s.hajizadeh/src/rnki/classification/pipeline/_run.sh 

##-----------!/bin/bash
######### works like  Docker attach 
####### sbatch --gres=gpu:1 --cpus-per-task=12 --mem=64G /bin/bash
####### singularity shell --nv /home/s.hajizadeh/docker_singularity_images/siamakg_23feb2022.sif
### cd /home/s.hajizadeh/src/rnki/classification/pipeline/
### python3 scoring.py ###> ./scoring.txt 2>&1 ##&

# pwd
# ls src/rnki/classification/saves
# nvidia-smi

#####################################################################################################################################################


# json file
# //  Left matches slidescore, right matches classes used in model, example: 
# // {
# //     "tumor": "model_tumor",
# //     "lymphocyte": "model_lymphocyte",
# //     "fibroblast": "model_fibroblast",
# //     "dcis": "model_tumor",
# //     "macrophage": "model_other",
# //     "red_blood": "model_red",
# //     "nerve": "model_other",
# //     "epithelial": "model_normal",
# //     "myoepithelial": "model_normal",
# //     "luminal": "model_normal",
# //     "endothelial": "model_normal",
# //     "fat": "model_other",
# //     "other": "model_other",
# //     "unspecified": "model_other",
# //     "ink": "model_other",
# //     "model_epi_endo_thelial": "model_normal"
# // }
