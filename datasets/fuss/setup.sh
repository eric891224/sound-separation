#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Define the ROOT_DIR variable which will hold all downloaded/generated data.
# Uncomment next line.
ROOT_DIR=/home/eegroup/ee50524/b08901169/s3prl_lab

if [ x${ROOT_DIR} == x ]; then
  echo "Please define ROOT_DIR variable inside `dirname $0`/setup.sh."
  exit 1
fi

# Download directory to download data into.
DOWNLOAD_DIR=${ROOT_DIR}/download

# This is the main directory where the fixed dev data will reside.
DEV_DATA_DIR=${ROOT_DIR}/fuss_dev

# The ssdata archive file is assumed to include a top folder named ssdata.
SSDATA_URL="https://zenodo.org/record/3743844/files/FUSS_ssdata.tar.gz"

# The archive file is assumed to include a top folder named ssdata_reverb.
SSDATA_REVERB_URL="https://zenodo.org/record/3743844/files/FUSS_ssdata_reverb.tar.gz"

# The fsd data archive file is assumed to include a top folder named fsd_data.
FSD_DATA_URL="https://zenodo.org/record/3743844/files/FUSS_fsd_data.tar.gz"

# The rir data archive file is assumed to include a top folder named rir_data.
RIR_DATA_URL="https://zenodo.org/record/3743844/files/FUSS_rir_data.tar.gz"

# Random seed to use for augmented data generation.
RANDOM_SEED=2020

# Number of train and validation examples to generate for data augmentation.
NUM_TRAIN=20000
NUM_VAL=1000

# This is the main directory where the single source files and room impulse
# responses that will be used in data augmentation will be downloaded.
RAW_DATA_DIR=${ROOT_DIR}/fuss_sources_and_rirs

# This is the main directory where the augmented data will reside.
# _${RANDOM_SEED} will be appended to this path, so multiple versions
# of augmented data can be generated by only changing the random seed.
AUG_DATA_DIR=${ROOT_DIR}/fuss_augment
