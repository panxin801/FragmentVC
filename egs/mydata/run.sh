#!/bin/bash

source path.sh
set -eu
set -x

start_stage=0
end_stage=0

wav_dir=$1
feat_dir=$2

if [ ${start_stage} -ge 0 -a ${end_stage} -le 0 ]; then
    bash local/preprocess.sh ${wav_dir} ${feat_dir} || exit 1
fi
