#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee

cd $PBS_O_WORKDIR
source activate b08901169_s3prl_env
module load cuda/cuda-10.0/x86_64

python3 -m \
	heareval.embeddings.runner \
	HearModels.hubert \
	--tasks-dir ./tasks/

conda deactivate
