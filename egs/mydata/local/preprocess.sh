#!/bin/bash

set -euo pipefail

src_dir=$1
wav2vec_path=./model_zoo/wav2vec_small.pt
feat_dir=$2

python ${SRC_ROOT}/data_prep.py \
    ${src_dir} \
    ${wav2vec_path} \
    ${feat_dir} \
    --audio-config config/audio.yaml
