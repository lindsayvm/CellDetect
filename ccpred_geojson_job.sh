#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=a6000
#SBATCH --nodelist ptolemaeus
#SBATCH --job-name=celldetect
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=slurm_output_%j.out
#SBATCH --cpus-per-task=16

JOBS_SOURCE="/home/l.leek/src/CellDetect/"
SINGULARITYIMAGE="/home/l.leek/docker_singularity_images/u20c114s.sif"
COMMAND="python3 export_ccpred_geojson.py"

singularity exec --nv \
--no-home \
--bind "$JOBS_SOURCE" \
--bind "$SCRATCH" \
--pwd "$JOBS_SOURCE" \
$SINGULARITYIMAGE \
$COMMAND 

echo "Job finished succesfully"

####### run it as $ sbatch run.sh, https://hpc-uit.readthedocs.io/en/latest/jobs/examples.html
###### --no-home ensures that nothing is used from outside the container
