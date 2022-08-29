#!/bin/bash

singularity exec --nv /path/to/singularity_image.sif \
    bash -c "cd /path/to/repo-root/classification/pipeline/ && python3  scoring_cupy_single.py $PWD $1"
