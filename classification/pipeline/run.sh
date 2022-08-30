#!/bin/bash
#ssh rhpc-eratosthenes  

singularity exec --nv /home/l.leek/docker_singularity_images/siamakg_23feb2022.sif \
    bash -c "cd /home/l.leek/src/CellDetect/classification/pipeline/ && python3  scoring_cupy_single.py $PWD $1"

#Traceback (most recent call last):
#  File "scoring_cupy_single.py", line 19, in <module>
#    from tracking_single import TaskTracker
#ModuleNotFoundError: No module named 'tracking_single'

cd src/CellDetect/
singularity exec --nv /home/l.leek/docker_singularity_images/siamakg_23feb2022.sif  \
    bash -c "python3 -m classification.pipeline.scoring_cupy_single classification/pipeline/run_example/"

cd src/CellDetect/ 
singularity exec --nv /home/l.leek/docker_singularity_images/siamakg_23feb2022.sif  \
    bash -c "python3 -m classification.pipeline.scoring_cupy_single /home/l.leek/src/CellDetect/IID/"




#Example
#cd HECellClassification/
#python3 -m classification.pipeline.scoring_cupy_single classification/pipeline/run_example/

#IID data
#cd HECellClassification/
#python3 -m classification.pipeline.scoring_cupy_single /data/lindsay/HECellClassification/IID/
