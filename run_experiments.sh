#! /bin/bash
set -e

ACQF="EEIPU"

for trial in {1..10..2}; do

    cache_root=.cachestore/${ACQF}/${RANDOM}_trial_${trial}
    CUDA_VISIBLE_DEVICES=0 python segmentation/experiments.py \
        --exp-name segmentation --trial $trial --cache-root \
        $cache_root --acqf $ACQF && rm -rf $cache_root &

    cache_root=.cachestore/${ACQF}/${RANDOM}_trial_$(($trial+1))
    CUDA_VISIBLE_DEVICES=1 python segmentation/experiments.py \
        --exp-name segmentation --trial $(($trial+1)) --cache-root \
        $cache_root --acqf $ACQF && rm -rf $cache_root
done

wait